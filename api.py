#!/usr/bin/env python3
"""
Quote Execution Quality Analyzer — REST API
============================================
Exposes execution metrics for any Hyperliquid fill group by wallet + tx hash.

Usage
-----
  pip install -r requirements.txt
  uvicorn api:app --host 0.0.0.0 --port 8080

Endpoints
---------
  POST /analyze   { "address": "0x...", "txHash": "0x..." }
  GET  /health
  GET  /docs      (Swagger UI, auto-generated)
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
HL_API    = "https://api.hyperliquid.xyz/info"
FEE_TAKER = 0.00045   # 4.5 bps
FEE_MAKER = 0.0001    # 1.0 bps
FEE_DELTA = FEE_TAKER - FEE_MAKER

# Optional: set to your HypeRPC orderbook key for historical book data.
# Format: "https://orderbook-eu.hyperpc.app/YOUR_KEY"
HYPERPC_ORDERBOOK: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Quote Execution Analyzer API",
    version="1.0.0",
    description=(
        "Compute execution quality metrics (execution score, implementation shortfall, "
        "VS period VWAP, all-in cost breakdown) for any Hyperliquid trade."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════
class AnalyzeRequest(BaseModel):
    address: str = Field(..., description="Hyperliquid wallet address (0x + 40 hex chars)")
    txHash: str  = Field(..., description="Fill action hash (0x + 64 hex chars)")


class CostComponent(BaseModel):
    bps:    float
    usd:    float
    source: str   # "exact" | "approx"
    note:   str


class AllInBreakdown(BaseModel):
    bps:      float
    usd:      float
    breakdown: dict[str, CostComponent]


class Metric(BaseModel):
    available: bool
    bps:       Optional[float] = None
    usd:       Optional[float] = None
    note:      str


class ExecScore(BaseModel):
    score: int          # 0–100
    grade: str          # A | B | C | D | F
    label: str          # Excellent / Good / Fair / Poor / Very Poor
    color: str          # hex colour for UI use


class AnalyzeResponse(BaseModel):
    # Identity
    asset:         str
    side:          str     # LONG | SHORT
    execTimeMs:    int
    # Trade facts (EXACT from fills)
    totalSize:     float
    totalNotional: float
    vwap:          float
    firstPx:       float
    lastPx:        float
    numFills:      int
    takerPct:      float
    durationSec:   float
    # Reference
    referencePrice:  float
    referenceSource: str   # "candleClose" | "firstFillApprox"
    # Market context
    liveSpreadBps: float
    dayVolUsd:     Optional[float]
    volParticipationPct: Optional[float]
    # ── KEY METRICS ──────────────────────────────────────────────
    execScore:               ExecScore
    implementationShortfall: Metric
    vsPeriodVwap:            Metric
    allInCost:               AllInBreakdown


# ══════════════════════════════════════════════════════════════════
#  HYPERLIQUID API HELPER
# ══════════════════════════════════════════════════════════════════
async def hl_post(body: dict) -> Any:
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            HL_API,
            json=body,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.json()


async def fetch_historical_book(coin: str, timestamp_ms: int) -> Optional[dict]:
    """Fetch trade-time orderbook via HypeRPC (requires HYPERPC_ORDERBOOK key)."""
    if not HYPERPC_ORDERBOOK:
        return None
    FORTY_DAYS = 40 * 24 * 3600 * 1000
    import time
    if (int(time.time() * 1000) - timestamp_ms) > FORTY_DAYS:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                HYPERPC_ORDERBOOK,
                json={"type": "l2Book", "coin": coin, "timestamp": timestamp_ms},
                headers={"Content-Type": "application/json"},
            )
            if not r.is_success:
                return None
            data = r.json()
            lvls = data.get("levels", [])
            if len(lvls) < 2 or not lvls[0] or not lvls[1]:
                return None
            return data
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
#  FILL HELPERS
# ══════════════════════════════════════════════════════════════════
def find_fills_by_hash(fills: list, tx_hash: str) -> list:
    clean = tx_hash.lower()
    return [f for f in fills if f.get("hash", "").lower() == clean]


def summarise_order(fills: list) -> dict:
    """Aggregate raw fills into an order summary. Matches JS logic exactly."""
    sorted_fills = sorted(fills, key=lambda f: int(f["time"]))
    total_sz  = sum(float(f["sz"]) for f in sorted_fills)
    total_ntl = sum(float(f["px"]) * float(f["sz"]) for f in sorted_fills)
    fees      = sum(abs(float(f.get("fee", 0))) for f in sorted_fills)

    # takerPct by notional weight (f.crossed == True means taker, matching JS)
    taker_ntl = sum(
        float(f["px"]) * float(f["sz"])
        for f in sorted_fills if f.get("crossed", False)
    )
    taker_pct = (taker_ntl / total_ntl * 100) if total_ntl > 0 else 0.0

    first_fill = sorted_fills[0]
    last_fill  = sorted_fills[-1]

    return {
        "coin":          first_fill["coin"],
        "side":          first_fill["side"],          # "B" or "A"
        "is_buy":        first_fill["side"] == "B",
        "total_sz":      total_sz,
        "total_ntl":     total_ntl,
        "vwap":          total_ntl / total_sz if total_sz else 0.0,
        "fees":          fees,
        "first_fill":    first_fill,
        "last_fill":     last_fill,
        "first_px":      float(first_fill["px"]),
        "last_px":       float(last_fill["px"]),
        "first_time":    int(first_fill["time"]),
        "time_span_ms":  int(last_fill["time"]) - int(first_fill["time"]),
        "taker_pct":     taker_pct,
        "num_fills":     len(sorted_fills),
        "hashes":        list({f.get("hash", "") for f in sorted_fills}),
    }


# ══════════════════════════════════════════════════════════════════
#  CANDLE HELPERS
# ══════════════════════════════════════════════════════════════════
async def fetch_candle_at(coin: str, time_ms: int) -> Optional[dict]:
    """Last fully-closed 1-min candle before time_ms (matches JS fetchCandleAt)."""
    candles = await hl_post({
        "type": "candleSnapshot",
        "req": {
            "coin":      coin,
            "interval":  "1m",
            "startTime": time_ms - 180_000,
            "endTime":   time_ms + 60_000,
        },
    })
    if not isinstance(candles, list) or not candles:
        return None
    # Must be FULLY closed before the trade: candle.T + 60000 <= time_ms
    before = [c for c in candles if c["T"] + 60_000 <= time_ms]
    c = before[-1] if before else candles[0]
    return {
        "open":  float(c["o"]),
        "high":  float(c["h"]),
        "low":   float(c["l"]),
        "close": float(c["c"]),
        "time":  c["T"],
    }


async def fetch_exec_candle(coin: str, trade_time: int) -> Optional[dict]:
    """The 1-min candle containing the trade (for VS Period VWAP)."""
    candles = await hl_post({
        "type": "candleSnapshot",
        "req": {
            "coin":      coin,
            "interval":  "1m",
            "startTime": trade_time - 60_000,
            "endTime":   trade_time + 60_000,
        },
    })
    if not isinstance(candles, list) or not candles:
        return None
    match = next(
        (c for c in candles if c["T"] <= trade_time < c["T"] + 60_000), None
    )
    c = match or candles[0]
    o, h, l, cl = float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"])
    return {
        "open":  o, "high": h, "low": l, "close": cl, "time": c["T"],
        "vwap":  (o + h + l + cl) / 4,   # OHLC4 = proxy for period average
    }


# ══════════════════════════════════════════════════════════════════
#  ORDERBOOK HELPERS
# ══════════════════════════════════════════════════════════════════
def get_spread(book: dict) -> dict:
    bids = book.get("levels", [[]])[0]
    asks = book.get("levels", [[], []])[1]
    if not bids or not asks:
        return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "abs": 0.0, "bps": 0.0}
    bid = float(bids[0]["px"])
    ask = float(asks[0]["px"])
    mid = (bid + ask) / 2
    return {
        "bid": bid, "ask": ask, "mid": mid,
        "abs": ask - bid,
        "bps": (ask - bid) / mid * 10_000 if mid else 0.0,
    }


def sim_market_order(book: dict, is_buy: bool, target_ntl: float) -> dict:
    """Walk the book to fill target_ntl, return VWAP and consumed levels."""
    levels = book.get("levels", [[], []])[1 if is_buy else 0]
    remaining  = target_ntl
    total_filled = 0.0
    total_size   = 0.0
    consumed = []
    for lvl in levels:
        if remaining < 1e-9:
            break
        px    = float(lvl["px"])
        sz    = float(lvl["sz"])
        avail = px * sz
        take  = min(remaining, avail)
        consumed.append({"px": px, "sz": take / px, "notional": take})
        total_filled += take
        total_size   += take / px
        remaining    -= take
    vwap = total_filled / total_size if total_size else 0.0
    return {
        "vwap":     vwap,
        "fillPct":  (total_filled / target_ntl * 100) if target_ntl else 0.0,
        "shortfall": remaining,
        "levels":   consumed,
    }


def sim_twap(book: dict, is_buy: bool, total_ntl: float, num_slices: int) -> dict:
    """Simulate one TWAP slice; total cost = same bps × full notional."""
    slice_ntl  = total_ntl / num_slices
    arrival_px = float(
        book["levels"][1][0]["px"] if is_buy else book["levels"][0][0]["px"]
    )
    sl = sim_market_order(book, is_buy, slice_ntl)
    slice_bps = abs(sl["vwap"] - arrival_px) / arrival_px * 10_000 if arrival_px else 0.0
    return {
        "numSlices":    num_slices,
        "sliceNtl":    slice_ntl,
        "sliceVwap":   sl["vwap"],
        "sliceBps":    slice_bps,
        "totalCostUsd": slice_bps / 10_000 * total_ntl,
    }


# ══════════════════════════════════════════════════════════════════
#  SCORE
# ══════════════════════════════════════════════════════════════════
def compute_exec_score(
    all_in_bps: float,
    taker_pct: float,
    vol_pct: Optional[float],
) -> dict:
    """0–100 numeric score + A-F grade. Matches JS computeExecScore."""
    score = max(0, round(100 - all_in_bps * 5))
    if taker_pct >= 95 and vol_pct is not None and vol_pct > 0.5:
        score = max(0, score - 15)
    if   score >= 90: label, grade, col = "Excellent", "A", "#34d399"
    elif score >= 80: label, grade, col = "Excellent", "A", "#34d399"
    elif score >= 75: label, grade, col = "Good",      "B", "#a3e635"
    elif score >= 60: label, grade, col = "Good",      "B", "#a3e635"
    elif score >= 50: label, grade, col = "Fair",      "C", "#fbbf24"
    elif score >= 40: label, grade, col = "Fair",      "C", "#fbbf24"
    elif score >= 25: label, grade, col = "Poor",      "D", "#fb923c"
    elif score >= 20: label, grade, col = "Poor",      "D", "#fb923c"
    else:             label, grade, col = "Very Poor", "F", "#f87171"
    return {"score": score, "grade": grade, "label": label, "color": col}


# ══════════════════════════════════════════════════════════════════
#  CORE METRIC COMPUTATION  (mirrors JS calcMetrics exactly)
# ══════════════════════════════════════════════════════════════════
def calc_metrics(
    order:       dict,
    book:        dict,
    candle:      Optional[dict],
    exec_candle: Optional[dict],
    asset_ctx:   Optional[dict],
) -> dict:
    is_buy    = order["is_buy"]
    sp        = get_spread(book)
    first_px  = order["first_px"]
    vwap      = order["vwap"]
    total_sz  = order["total_sz"]
    total_ntl = order["total_ntl"]

    # ── Reference price ───────────────────────────────────────────
    if candle:
        ref_price  = candle["close"]
        ref_source = "candleClose"
    else:
        ref_price  = first_px - sp["abs"] / 2 if is_buy else first_px + sp["abs"] / 2
        ref_source = "firstFillApprox"

    ref_ntl = ref_price * total_sz

    # ── EXACT metrics ─────────────────────────────────────────────
    # Book-walk: VWAP vs first fill, denominated by refPrice for additivity
    walk_per_unit = abs(vwap - first_px)
    walk_usd      = walk_per_unit * total_sz
    walk_bps      = (walk_per_unit / ref_price) * 10_000 if ref_price else 0.0

    # Fees: exact from fills
    fee_usd = order["fees"]
    fee_bps = (fee_usd / ref_ntl) * 10_000 if ref_ntl else FEE_TAKER * 10_000

    # ── APPROXIMATE metrics ───────────────────────────────────────
    # Implementation Shortfall: VWAP vs reference price
    slip_per_unit = (vwap - ref_price) if is_buy else (ref_price - vwap)
    slip_usd      = slip_per_unit * total_sz
    slip_bps      = slip_per_unit / ref_price * 10_000 if ref_price else 0.0

    # Spread cost: (2×takerPct − 1) × halfSpread
    half_sp_bps = sp["bps"] / 2
    spread_bps  = (2 * order["taker_pct"] / 100 - 1) * half_sp_bps
    spread_usd  = spread_bps / 10_000 * ref_ntl

    # Pre-fill drift: firstFill premium over ref, minus the spread component
    first_fill_premium = (first_px - ref_price) if is_buy else (ref_price - first_px)
    drift_bps          = first_fill_premium / ref_price * 10_000 - spread_bps if ref_price else 0.0
    drift_usd          = drift_bps / 10_000 * ref_ntl

    # All-in cost
    all_in_usd = slip_usd + fee_usd
    all_in_bps = (all_in_usd / ref_ntl) * 10_000 if ref_ntl else 0.0

    # ── VS Period VWAP ────────────────────────────────────────────
    vs_vwap_bps = vs_vwap_usd = period_vwap = None
    if exec_candle:
        period_vwap  = exec_candle["vwap"]
        vs_per_unit  = (vwap - period_vwap) if is_buy else (period_vwap - vwap)
        vs_vwap_bps  = vs_per_unit / period_vwap * 10_000 if period_vwap else 0.0
        vs_vwap_usd  = vs_vwap_bps / 10_000 * ref_ntl

    # ── Market context ────────────────────────────────────────────
    day_vol = float(asset_ctx.get("dayNtlVlm", 0)) if asset_ctx else 0.0
    vol_pct = (total_ntl / day_vol * 100) if day_vol > 0 else None

    exec_score = compute_exec_score(all_in_bps, order["taker_pct"], vol_pct)

    return {
        # Identity
        "asset":   order["coin"],
        "side":    "LONG" if is_buy else "SHORT",
        "execTimeMs": order["first_time"],
        # Trade facts
        "totalSize":     round(total_sz, 8),
        "totalNotional": round(total_ntl, 2),
        "vwap":          round(vwap, 6),
        "firstPx":       round(first_px, 6),
        "lastPx":        round(order["last_px"], 6),
        "numFills":      order["num_fills"],
        "takerPct":      round(order["taker_pct"], 2),
        "durationSec":   order["time_span_ms"] / 1000,
        # Reference
        "referencePrice":  round(ref_price, 6),
        "referenceSource": ref_source,
        # Market context
        "liveSpreadBps": round(sp["bps"], 3),
        "dayVolUsd":     round(day_vol, 0) if day_vol else None,
        "volParticipationPct": round(vol_pct, 4) if vol_pct is not None else None,
        # ── KEY METRICS ──────────────────────────────────────────
        "execScore": exec_score,
        "implementationShortfall": {
            "available": True,
            "bps":  round(slip_bps, 2),
            "usd":  round(slip_usd, 2),
            "note": (
                f"Fill VWAP ({vwap:.4f}) vs "
                f"{'1-min candle close' if candle else 'estimated mid'} "
                f"({ref_price:.4f})"
            ),
        },
        "vsPeriodVwap": {
            "available": exec_candle is not None,
            "bps":  round(vs_vwap_bps, 2) if vs_vwap_bps is not None else None,
            "usd":  round(vs_vwap_usd, 2) if vs_vwap_usd is not None else None,
            "periodVwap": round(period_vwap, 6) if period_vwap else None,
            "note": (
                "Fill VWAP vs OHLC4 of execution candle"
                if exec_candle else "Execution candle unavailable"
            ),
        },
        "allInCost": {
            "bps": round(all_in_bps, 2),
            "usd": round(all_in_usd, 2),
            "breakdown": {
                "drift": {
                    "bps":    round(drift_bps, 2),
                    "usd":    round(drift_usd, 2),
                    "source": "approx",
                    "note":   (
                        f"Price moved ref→firstFill: "
                        f"{ref_price:.4f} → {first_px:.4f}"
                    ),
                },
                "spread": {
                    "bps":    round(spread_bps, 2),
                    "usd":    round(spread_usd, 2),
                    "source": "approx",
                    "note":   (
                        f"{'Taker: paid' if order['taker_pct'] > 95 else 'Maker: received' if order['taker_pct'] < 5 else 'Mixed:'}"
                        f" ½ live spread ({half_sp_bps:.2f} bps)"
                    ),
                },
                "bookWalk": {
                    "bps":    round(walk_bps, 2),
                    "usd":    round(walk_usd, 2),
                    "source": "exact",
                    "note":   (
                        f"VWAP ({vwap:.4f}) vs firstFill ({first_px:.4f}) "
                        f"— exact from fills"
                    ),
                },
                "fees": {
                    "bps":    round(fee_bps, 2),
                    "usd":    round(fee_usd, 2),
                    "source": "exact",
                    "note":   f"From fill data: {fee_usd / total_ntl * 100:.4f}% of notional",
                },
            },
        },
    }


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════
@app.get("/health", tags=["meta"])
async def health():
    """Liveness check."""
    return {"status": "ok", "service": "quote-exec-analyzer"}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Compute execution quality metrics for a Hyperliquid trade",
    description="""
Provide a wallet address and the fill action hash (from the Hyperliquid explorer).
The endpoint fetches all on-chain fills for that hash, derives the full execution
profile, and returns:

- **execScore** — letter grade (A–F) + numeric score (0–100)
- **implementationShortfall** — fill VWAP vs pre-trade 1-min candle close
- **vsPeriodVwap** — fill VWAP vs OHLC4 of the execution-window candle
- **allInCost** — total execution cost with breakdown (drift / spread / book-walk / fees)

`source: "exact"` fields are computed directly from fill data.
`source: "approx"` fields require the candle-based reference price.
""",
)
async def analyze(req: AnalyzeRequest):
    # ── Validate ──────────────────────────────────────────────────
    if not re.match(r"^0x[0-9a-fA-F]{40}$", req.address):
        raise HTTPException(400, "Invalid address — must be 0x + 40 hex chars")
    if not re.match(r"^0x[0-9a-fA-F]{64}$", req.txHash):
        raise HTTPException(400, "Invalid txHash — must be 0x + 64 hex chars")

    # ── 1. Fetch all fills ────────────────────────────────────────
    all_fills = await hl_post({"type": "userFills", "user": req.address})
    if not isinstance(all_fills, list) or not all_fills:
        raise HTTPException(404, "No fills found for this address")

    # ── 2. Match by tx hash ───────────────────────────────────────
    matched = find_fills_by_hash(all_fills, req.txHash)
    if not matched:
        raise HTTPException(
            404,
            f"No fills matched txHash {req.txHash}. "
            "Verify the address and hash both belong to the same order.",
        )

    order    = summarise_order(matched)
    coin     = order["coin"]
    trade_ms = order["first_time"]

    # ── 3. Parallel data fetch ────────────────────────────────────
    hist_book, candle, exec_candle, live_book, meta = await asyncio.gather(
        fetch_historical_book(coin, trade_ms),
        fetch_candle_at(coin, trade_ms),
        fetch_exec_candle(coin, trade_ms),
        hl_post({"type": "l2Book", "coin": coin}),
        hl_post({"type": "metaAndAssetCtxs"}),
    )

    # Use historical book if available, else live snapshot
    book = hist_book if hist_book else live_book

    # ── 4. Extract asset context ──────────────────────────────────
    asset_ctx = None
    if isinstance(meta, list) and len(meta) == 2:
        universe = meta[0].get("universe", [])
        ctx_list = meta[1]
        for i, u in enumerate(universe):
            if u.get("name") == coin and i < len(ctx_list):
                asset_ctx = ctx_list[i]
                break

    # ── 5. Compute and return ─────────────────────────────────────
    return calc_metrics(order, book, candle, exec_candle, asset_ctx)
