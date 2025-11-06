import pandas as pd
from datetime import datetime
import streamlit as st


def ensure_state(st):
    st.session_state.setdefault("cash", float(st.session_state.get("cash_balance", 0.0)))
    st.session_state.setdefault("realized_pl", 0.0)
    st.session_state.setdefault("transactions", [])
    st.session_state.setdefault("portfolio_history", [])
    st.session_state.setdefault("latest_stock_prices", {})
    st.session_state.setdefault("latest_fund_prices", {})

def _normalize_lot(lot: dict):
    if not isinstance(lot, dict):
        return 0.0, 0.0
    qty = (lot.get("qty") or lot.get("quantity") or lot.get("units")
           or lot.get("shares") or lot.get("amount") or 0.0)
    avg = (lot.get("avg_cost") or lot.get("avg_price")
           or lot.get("average_price") or lot.get("avg")
           or lot.get("price") or 0.0)
    try: qty = float(qty)
    except: qty = 0.0
    try: avg = float(avg)
    except: avg = 0.0
    return qty, avg

def _positions_from_transactions(st):
    """
    Build {class:{symbol:{qty, avg_cost}}} from st.session_state['transactions'].
    Uses moving-average cost: avg_cost updates on BUY only; SELL reduces qty.
    """
    pos = {"stocks": {}, "funds": {}}
    txns = st.session_state.get("transactions", [])
    if not txns:
        return pos
    for t in txns:
        cls = str(t.get("asset_class", "")).lower()  # 'stocks' or 'funds'
        if cls not in pos:
            continue
        sym = str(t.get("symbol"))
        side = str(t.get("side", "")).lower()
        qty  = float(t.get("qty", 0.0) or 0.0)
        px   = float(t.get("price", 0.0) or 0.0)
        lot = pos[cls].setdefault(sym, {"qty": 0.0, "avg_cost": 0.0})

        if side == "buy":
            # moving-average update
            new_qty = lot["qty"] + qty
            if new_qty > 0:
                lot["avg_cost"] = (lot["avg_cost"] * lot["qty"] + px * qty) / new_qty
            lot["qty"] = new_qty
        elif side == "sell":
            lot["qty"] = max(lot["qty"] - qty, 0.0)  # avg_cost unchanged on sell

    # drop zero positions
    for cls in list(pos.keys()):
        pos[cls] = {s: l for s, l in pos[cls].items() if l["qty"] > 0}
    return pos

def _merged_holdings(st):
    """
    Prefer explicit app dicts; if missing/empty for a class, fall back to transactions.
    """
    stocks, funds = {}, {}

    # 1) Try canonical + common app keys
    h = st.session_state.get("holdings")
    if isinstance(h, dict):
        stocks.update(h.get("stocks", {}) or {})
        funds.update(h.get("funds", {}) or {})

    for k in ["stock_holdings", "stocks_holdings", "stock_positions",
              "positions_stocks", "equity_holdings", "equity_positions"]:
        d = st.session_state.get(k)
        if isinstance(d, dict):
            for sym, lot in d.items():
                q, a = _normalize_lot(lot)
                if q > 0:
                    stocks[sym] = {"qty": q, "avg_cost": a}

    for k in ["fund_holdings", "fund_positions", "funds_holdings"]:
        d = st.session_state.get(k)
        if isinstance(d, dict):
            for sym, lot in d.items():
                q, a = _normalize_lot(lot)
                if q > 0:
                    funds[sym] = {"qty": q, "avg_cost": a}

    # 2) If a class is still empty, reconstruct it from transactions
    tx_pos = _positions_from_transactions(st)
    if not stocks and tx_pos["stocks"]:
        stocks = tx_pos["stocks"]
    if not funds and tx_pos["funds"]:
        funds = tx_pos["funds"]

    return {"stocks": stocks, "funds": funds}

def get_live_price(symbol: str, asset_class: str) -> float:
    if asset_class == "stocks":
        return float(st.session_state.get("latest_stock_prices", {}).get(symbol, 0.0))
    return float(st.session_state.get("latest_fund_prices", {}).get(symbol, 0.0))

def compute_holdings_df(st) -> pd.DataFrame:
    rows = []
    merged = _merged_holdings(st)

    for asset_class in ("stocks", "funds"):
        for symbol, lot in (merged.get(asset_class) or {}).items():
            qty, avg_cost = _normalize_lot(lot)
            if qty <= 0:
                continue
            last = get_live_price(symbol, asset_class)
            value = qty * last
            cost = qty * avg_cost
            unreal_pl = value - cost
            unreal_pct = (unreal_pl / cost * 100.0) if cost else 0.0
            rows.append({
                "Asset": symbol,
                "Class": asset_class.capitalize(),
                "Qty": qty,
                "Avg Cost": round(avg_cost, 4),
                "Last Price": round(last, 4),
                "Value": round(value, 2),
                "Unrealized P/L": round(unreal_pl, 2),
                "Unrealized %": round(unreal_pct, 2),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "Asset","Class","Qty","Avg Cost","Last Price","Value","Unrealized P/L","Unrealized %"
        ])
    return pd.DataFrame(rows)

def compute_totals(st, holdings_df: pd.DataFrame):
    portfolio_value = float(holdings_df["Value"].sum()) if not holdings_df.empty else 0.0
    unrealized_pl = float(holdings_df["Unrealized P/L"].sum()) if not holdings_df.empty else 0.0
    realized_pl = float(st.session_state.get("realized_pl", 0.0))
    total_profit = realized_pl + unrealized_pl
    cash = float(st.session_state.get("cash", st.session_state.get("cash_balance", 0.0)))
    total_value = cash + portfolio_value
    return {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "realized_pl": realized_pl,
        "unrealized_pl": unrealized_pl,
        "total_profit": total_profit,
        "total_value": total_value,
    }

def record_txn(st, *, side, asset_class, symbol, qty, price, fee=0.0, ts=None):
    ensure_state(st)
    ts = ts or datetime.now()
    st.session_state["transactions"].append({
        "ts": ts, "side": side, "asset_class": asset_class, "symbol": symbol,
        "qty": float(qty), "price": float(price), "fee": float(fee),
        "cash_after": float(st.session_state.get("cash_balance", st.session_state.get("cash", 0.0))),
    })
    snapshot_portfolio_history(st, ts=ts)

def snapshot_portfolio_history(st, ts=None):
    ts = ts or datetime.now()
    holdings_df = compute_holdings_df(st)
    totals = compute_totals(st, holdings_df)
    st.session_state["portfolio_history"].append({
        "ts": ts,
        "cash": totals["cash"],
        "portfolio_value": totals["portfolio_value"],
        "total_value": totals["total_value"],
        "realized_pl": totals["realized_pl"],
        "unrealized_pl": totals["unrealized_pl"],
    })