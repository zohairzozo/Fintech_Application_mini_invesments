import yfinance as yf
import streamlit as st
from fund_data import load_fund_data  # <-- use your real-data loader

# Load once and reuse
FUND_DATA = load_fund_data()


def _get_latest_fund_price(fund_name: str) -> float | None:
    """
    Get the latest NAV for a fund from FUND_DATA.
    Expects each entry in FUND_DATA[fund_name] to have either:
      - 'latest_nav', or
      - 'nav' DataFrame with 'navPerShare' column.
    """
    fd = FUND_DATA.get(fund_name)
    if not fd:
        return None

    # Preferred: explicit latest_nav
    latest_nav = fd.get("latest_nav")
    if latest_nav is not None:
        try:
            return float(latest_nav)
        except Exception:
            pass

    # Fallback: last row of nav timeseries
    nav_df = fd.get("nav")
    if nav_df is not None and not nav_df.empty and "navPerShare" in nav_df.columns:
        try:
            return float(nav_df["navPerShare"].iloc[-1])
        except Exception:
            pass

    return None


def _recompute_equity_value():
    """
    Recalculate TOTAL equity from current holdings (stocks + funds) using recent prices.
    - Stocks: yfinance (fallback to last_price)
    - Funds: Excel-based FUND_DATA latest NAV (fallback to last_price)
    Updates: st.session_state.equity_value
    Also refreshes 'last_price' for each holding.
    """
    import pandas as pd  # if you need it elsewhere; safe to keep

    total = 0.0

    # ---------- STOCKS ----------
    holdings = st.session_state.get("holdings", {})
    for sym, pos in holdings.items():
        qty = float(pos.get("qty", 0.0))

        if qty <= 0:
            continue

        # Try to download recent data; handle cases where download returns None
        try:
            hist = yf.download(sym, period="5d", auto_adjust=False, progress=False)
        except Exception:
            hist = None

        # Default fallback price from cached last_price
        px = float(pos.get("last_price", 0.0) or 0.0)

        # Use adjusted close if available and non-empty
        if hist is not None and "Adj Close" in getattr(hist, "columns", []):
            try:
                adj = hist["Adj Close"].dropna()
                if not adj.empty:
                    px = float(adj.iloc[-1])
            except Exception:
                # keep px as fallback
                pass

        pos["last_price"] = px  # cache
        total += qty * px

    # ---------- FUNDS ----------
    fund_holdings = st.session_state.get("fund_holdings", {})
    for fname, fpos in fund_holdings.items():
        qty = float(fpos.get("qty", 0.0))

        if qty <= 0:
            continue

        fpx = _get_latest_fund_price(fname)
        if fpx is None:
            fpx = float(fpos.get("last_price", 0.0) or 0.0)

        fpos["last_price"] = fpx  # cache
        total += qty * fpx

    st.session_state.equity_value = round(float(total), 2)
