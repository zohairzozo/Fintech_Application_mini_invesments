# app.py
import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as pgo
import altair as alt

from stock_buy import _buy
from stock_sell import _sell
from fake_fund_data import _fake_fund_data as fund_data
from equity import _recompute_equity_value
from fund_buy import _fund_buy
from fund_sell import _fund_sell
from state_helpers import (
    ensure_state, #ok
    record_txn,
    get_live_price,
    compute_holdings_df,
    compute_totals,
    snapshot_portfolio_history,
    _normalize_lot, #ok
    _merged_holdings,
    _positions_from_transactions #ok

)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# -------------------------
# App setup & global styles
# -------------------------
st.set_page_config(
    page_title="Student Investment Portfolio",
    page_icon="📈",
    layout="wide"
)

CARD_CSS = """
<style>
/* Layout helpers */
.first-page {
  min-height: 12.5vh;
  display: grid;
  place-items: center;
}

/* Generic Card */
.card {
  background: white;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 6px 18px -10px rgba(0,0,0,0.25);
  text-align: center;
}

/* Headings inside cards */
.card h1, .card h2 { color:#111827; }

/* Paragraph centering helper */
.center-text {
  margin: 12px 0 22px 0;
  color: #4b5563;
  font-size: 1.05rem;
  line-height: 1.6;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

/* KPI Cards (single-block HTML render to avoid nesting issues) */
.kpi-card{
  background-color:#3b82f6 !important;  /* blue card */
  border:1px solid rgba(0,0,0,0.08);
  border-radius:16px;
  padding:16px 18px;
  box-shadow:0 6px 18px -10px rgba(0,0,0,0.25);
  color:#ffffff !important;
}
.kpi-title{
  margin:0 0 6px 0;
  font-weight:700;
  font-size:1rem;
  color:#ffffff !important;
}
.kpi-value{
  margin:0;
  font-size:1.6rem;
  font-weight:800;
  color:#ffffff !important;
}

div[data-testid="stHorizontalBlock"] div.option-card,
div[data-testid="stVerticalBlock"] div.option-card,
.option-card {
  background-color:#3b82f6 !important;   /* force blue background */
  border:1px solid rgba(0,0,0,0.08);
  border-radius:16px;
  padding:10px;
  box-shadow:0 6px 18px -10px rgba(0,0,0,0.25);
  color:#ffffff !important;
}
.option-title{
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 6px;
  margin-top: 0;
  color: #111827;
  
}

/* Better contrast when user uses dark theme */
@media (prefers-color-scheme: dark){
  .card, .option-card{ background:#111827 !important; border-color:rgba(255,255,255,0.08); }
  .card h1, .card h2, .option-title{ color:#f3f4f6; }
  .center-text, .option-help{ color:#d1d5db; }
}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# -------------------------
# Session state management
# -------------------------
def reset_portfolio():
    st.session_state.portfolio_name = ""
    st.session_state.cash_balance = 5000.00   # Hard-coded starting balance per spec
    st.session_state.equity_value = 0.00
    st.session_state.profit_value = 0.00
    st.session_state.selected_stock = None
    st.session_state.selected_fund = None

def ss_init():
    ss = st.session_state
    defaults = {
        "page": "page1",                # page1 -> page2 -> page3 -> page4
        "portfolio_name": "",
        "cash_balance": 5000.00,        # start at €5000
        "equity_value": 0.00,
        "profit_value": 0.00,
        "selected_stock": None,
        "selected_fund": None,
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

ss_init()

# --- Trading constants (hard-coded fees per BRD summary page)
BUY_FEE  = 1.00   # € flat
SELL_FEE = 1.00   # € flat

if "cash_balance" not in st.session_state:
    st.session_state.cash_balance = 5000.00
    
# --- Minimal portfolio state
if "holdings" not in st.session_state:
    # structure: { "AAPL": {"qty": float, "avg_cost": float, "last_price": float} }
    st.session_state.holdings = {}

if "txns" not in st.session_state:
    # each: {"ts": datetime, "side": "BUY"/"SELL", "symbol": str, "qty": int, "price": float, "fee": float}
    st.session_state.txns = []
    
if "profit_value" not in st.session_state:
    st.session_state.profit_value = 0.0
    
if "sell_view_cache" not in st.session_state:
    st.session_state.sell_view_cache = None
    
# --- Funds session state (add once) ---
if "fund_holdings" not in st.session_state:
    # { "OP-Rohkea A": {"qty": float, "avg_cost": float, "last_price": float} }
    st.session_state.fund_holdings = {}


def _build_sell_table_all():
    """
    Returns a DataFrame with both STOCK and FUND positions:
    columns: Type, Symbol, Quantity, Avg Cost, Last Price, Market Value, Unrealized P/L
    """
    import pandas as pd

    rows = []

    # --- STOCKS ---
    holdings = st.session_state.get("holdings", {})
    if holdings:
        symbols = list(holdings.keys())
        px_map = {}
        try:
            data = yf.download(tickers=" ".join(symbols), period="5d", auto_adjust=False, group_by='ticker', threads=False)
            for sym in symbols:
                try:
                    if len(symbols) == 1 and "Adj Close" in data:
                        px = float(data["Adj Close"].dropna().iloc[-1])
                    else:
                        px = float(data[sym]["Adj Close"].dropna().iloc[-1])
                except Exception:
                    px = float(holdings[sym].get("last_price", 0.0))
                px_map[sym] = px
        except Exception:
            for sym in symbols:
                px_map[sym] = float(holdings[sym].get("last_price", 0.0))

        for sym, pos in holdings.items():
            qty = float(pos.get("qty", 0.0))
            if qty <= 0:
                continue
            avg = float(pos.get("avg_cost", 0.0))
            last_px = float(px_map.get(sym, pos.get("last_price", 0.0)))
            mval = qty * last_px
            upl  = (last_px - avg) * qty
            rows.append({
                "Type": "STOCK",
                "Symbol": sym,
                "Quantity": int(qty) if float(qty).is_integer() else qty,
                "Avg Cost": round(avg, 4),
                "Last Price": round(last_px, 4),
                "Market Value": round(mval, 2),
                "Unrealized P/L": round(upl, 2),
            })

    # --- FUNDS ---
    fund_hold = st.session_state.get("fund_holdings", {})
    fund_price_map = {}
    try:
        FUNDS_DATA = fund_data()  # you imported _fake_fund_data as fund_data()
        for fname, df in FUNDS_DATA.items():
            try:
                fund_price_map[fname] = float(df["Price"].iloc[-1])
            except Exception:
                pass
    except Exception:
        pass

    for fname, pos in fund_hold.items():
        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            continue
        avg = float(pos.get("avg_cost", 0.0))
        last_px = float(fund_price_map.get(fname, pos.get("last_price", 0.0)))
        mval = qty * last_px
        upl  = (last_px - avg) * qty
        rows.append({
            "Type": "FUND",
            "Symbol": fname,
            "Quantity": round(qty, 4),
            "Avg Cost": round(avg, 4),
            "Last Price": round(last_px, 4),
            "Market Value": round(mval, 2),
            "Unrealized P/L": round(upl, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Type", "Symbol"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["Type","Symbol","Quantity","Avg Cost","Last Price","Market Value","Unrealized P/L"])
    return df

    
def go(to_page: str):
    st.session_state.page = to_page
    st.rerun()

# -------------------------
# Page 1 — Create New Portfolio
# -------------------------
def render_page1():
    with st.container():
        st.markdown('<div class="first-page">', unsafe_allow_html=True)
        col = st.columns([1, 2, 1])[1]
        with col:
            st.markdown(
                """
                <div class="card">
                  <h1>Create a New Investment Portfolio</h1>
                  <p class="center-text">
                    Start a fresh portfolio for the simulation.<br>You’ll name it on the next screen.
                  </p>
                """,
                unsafe_allow_html=True,
            )
            create = st.button("🚀 Create a New Investment Portfolio", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)  # close .card
        st.markdown('</div>', unsafe_allow_html=True)  # close .first-page

    if create:
        reset_portfolio()
        go("page2")

# -------------------------
# Page 2 — Name the Portfolio
# -------------------------
def render_page2():
    st.markdown("### Name Your Investment Portfolio")
    st.write("")
    name = st.text_input("Name for the Investment Portfolio", value=st.session_state.portfolio_name, placeholder="e.g., My Future Fund")

    cols = st.columns([1, 1])
    with cols[0]:
        back = st.button("← Back", use_container_width=True)
    with cols[1]:
        cont = st.button("Continue →", type="primary", use_container_width=True, disabled=(len(name.strip()) == 0))

    if back:
        go("page1")
    if cont:
        st.session_state.portfolio_name = name.strip()
        go("page3")

# -------------------------
# Page 3 — Portfolio Overview
# -------------------------
def render_page3():
    pname = st.session_state.portfolio_name or "Unnamed Portfolio"
    st.markdown(f"## {pname}")

    kpi1, kpi2, kpi3 = st.columns([1.2, 1.2, 1.2])

    with kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Current Amount</div>
              <div class="kpi-value">€{st.session_state.cash_balance:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Equity Value</div>
              <div class="kpi-value">€{st.session_state.equity_value:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Equity / Profit</div>
              <div class="kpi-value">€{st.session_state.profit_value:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.divider()

    left, spacer = st.columns([0.35, 0.65])
    with left:
        st.markdown("#### Next step")
        st.markdown(
            """
            <p class="option-help">
              Continue to <strong>Investment Options</strong> to pick Stocks or Funds.
            </p>
            """,
            unsafe_allow_html=True,
        )
        # Now enabled: go to Page 4
        if st.button("📂 Investment Options", use_container_width=True):
            go("page4")

    with spacer:
        st.info(
            "Tip: These numbers are placeholders until you add investments. "
            "Next we’ll connect Stock Market & Investment Funds.",
            icon="💡",
        )

    st.write("")
    st.divider()
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("← Rename Portfolio", use_container_width=True):
            go("page2")
    with cols[1]:
        if st.button("Start Over", use_container_width=True):
            go("page1")

# -------------------------
# Page 4 — Investment Options (no subpages yet)
# -------------------------
STOCK_MAP = {
    "NVIDIA (NVDA)": "NVDA",
    "Apple (AAPL)": "AAPL",
    "Alphabet (GOOGL)": "GOOGL",
    "Puuilo (PUUILO.HE)": "PUUILO.HE",
    "Finnair (FIA1S.HE)": "FIA1S.HE",
    "Beyond Meat (BYND)": "BYND",
    "H&M (HM-B.ST)": "HM-B.ST",
    "Kone Oyj (KNEBV.HE)": "KNEBV.HE",
    "Neste Oyj (NESTE.HE)": "NESTE.HE",
    "Remedy Entertainment (REMEDY.HE)": "REMEDY.HE",
    "Nestlé (NESN.SW)": "NESN.SW",
    "Danske Bank (DANSKE.CO)": "DANSKE.CO",
    "Volkswagen (VOW3.DE)": "VOW3.DE",
    "Nordea (NDA-FI.HE)": "NDA-FI.HE",
}

FUND_LIST = [
    "OP-Rohkea A", "OP-Kehittyvät markkinat A", "OP-Suomi Indeksi A",
    "OP-Kiina A", "OP-Maailma A"
]

def render_page4():
    st.markdown("## Investment Options")

    st.caption("Choose either **Stock Market** or **Investment Funds** below.")

    col_left, col_center, col_right= st.columns(3)

    # ---- Stock Market box
    with col_left:
        st.markdown(
            f"""
            <div class="option-card">
                <div class="option-title center-text">Stock Market</div>
            </div>
            """,unsafe_allow_html=True)
        st.write("")
        
        stock_label = st.selectbox(
            "Select a stock",
            options=["— Select —"] + list(STOCK_MAP.keys()),  
            index=0,
            key="sb_stock"
        )
        if stock_label == "— Select —":
            st.session_state.selected_stock = None
        else:
            st.session_state.selected_stock = stock_label

        if st.button("Open Stock Page →", disabled=(st.session_state.selected_stock is None),
                      help="Stock details page with historical data, forecasts, and Buy/Sell."):
            st.session_state.page = "stock_subpage"
            st.rerun()


        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Investment Funds box
    with col_center:
        st.markdown(
            f"""
            <div class="option-card">
            <div class="option-title center-text">Investment Funds</div>
            </div>
            """,unsafe_allow_html=True)
        st.write("")    
        fund = st.selectbox(
            "Select a fund",
            options=["— Select —"] + FUND_LIST,
            index=0,
            key="sb_fund"
        )
        st.session_state.selected_fund = None if fund == "— Select —" else fund

        if st.button("Open Fund Page →", disabled=(st.session_state.selected_fund is None),
                  help="Fund details page with Morningstar stars, volatility, expenses, and Buy/Sell."):
            st.session_state.page = "fund_subpage"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_right:
        st.markdown(
            f"""
            <div class="option-card">
                <div class="option-title center-text">Sell Dahsboard</div>
            </div>
            """,unsafe_allow_html=True)
        st.write("")
        
        if st.button("Open Sell Dashboard →"):
            st.session_state.page = "sell_subpage"
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.divider()
    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("← Back to Overview", use_container_width=True):
            go("page3")
    with cols[1]:
        st.empty()
    with cols[2]:
        if st.button("Summary Page →", use_container_width=True):
            st.session_state.page = "summary_page"
            st.rerun()
        
def render_stock_subpage1():
    label = st.session_state.get("selected_stock")
    if not label or label not in STOCK_MAP:
        st.warning("Pick a stock on Investment Options first.")
        if st.button("← Back to Investment Options"):
            st.session_state.page = "page4"
            st.rerun()
        return

    symbol = STOCK_MAP[label]
    
    st.markdown(f"## {label} — Stock Details")

    # ----- Date range (default 1 year)
    today = dt.date.today()
    default_start = today - dt.timedelta(days=365)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", default_start, max_value=today)
    with c2:
        end_date = st.date_input("End date", today, min_value=start_date, max_value=today)

    # ----- Fetch data (restrict to whitelist symbol)
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust= False)
    if df.empty:
        st.error("❌ No data for this period. Try a different date range.")
        st.stop()

    # Flatten multiindex (dividends can add columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # keep only 'Close', 'Adj Close', etc.


    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        else:
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numcols:
                st.error("Yahoo returned no numeric price columns.")
                st.stop()
            df["Adj Close"] = df[numcols[0]]
            
    # ----- Aspects B: KPIs + base price chart
    # yfinance info
    info = {}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}

    # KPIs
    latest_price = float(df["Adj Close"].dropna().iloc[-1])
    first_price = float(df["Adj Close"].dropna().iloc[0])
    roi_pct = ((latest_price - first_price) / first_price) * 100 if first_price else np.nan

    pe = info.get("trailingPE", np.nan)
    div_per_share = info.get("dividendRate", np.nan)
    div_yield_pct = (info.get("dividendYield", np.nan) or np.nan) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Price / Earnings (trailing)", f"{pe:.2f}" if pd.notna(pe) else "—")
    k2.metric("ROI since start date", f"{roi_pct:.2f}%")
    k3.metric("Dividend / share", f"{div_per_share:.2f}" if pd.notna(div_per_share) else "—")
    k4.metric("Dividend yield", f"{div_yield_pct:.2f}%" if pd.notna(div_yield_pct) else "—")

    # Base chart: price trend for all variables (like your GitHub view)
    st.subheader("Stock price trend — all variables")
    data_show = df.copy().reset_index()
    numeric_cols = data_show.select_dtypes(include=np.number).columns
    fig_all = px.line(data_show, x="Date", y=numeric_cols, title="Stock Price Trends", width=1200, height=500)
    st.plotly_chart(fig_all, use_container_width=True)

    st.divider()
    st.markdown("**Advanced Analytics** (optional)")

    # ----- Aspects A in an expander (from your GitHub app)
    with st.expander("🔬 Open Advanced Analytics (SARIMAX, decomposition, forecast)"):
        # Select variable for modeling (like your GitHub `selectbox`)
        # Keep Date for plotting; subset to selected column

        column = st.selectbox("Select column for forecasting", numeric_cols, index=0)

        # Prepare data (as in GitHub)
        data_mod = df[[column]].copy()
        data_mod.insert(0, "Date", df.index)
        data_mod.reset_index(drop=True, inplace=True)
        st.write("Selected data", data_mod)

        # ADF stationarity test (like GitHub)
        st.markdown("<p style='color:green; font-style: italic;'>Is data Stationary?</p>", unsafe_allow_html=True)
        if len(data_mod[column]) < 15:
            st.warning("⚠️ Not enough data for stationarity test.")
        else:
            st.write(adfuller(data_mod[column])[1] < 0.05)

        # Decomposition (period ~ 12 for monthly-ish seasonality; your GitHub uses 12)
        st.subheader("Decomposition of Data")
        try:
            decomposition = seasonal_decompose(data_mod[column], model="additive", period=12)
            st.plotly_chart(px.line(x=data_mod["Date"], y=decomposition.trend, title="Trend", width=1200, height=400).update_traces(line_color="Green"), use_container_width=True)
            st.plotly_chart(px.line(x=data_mod["Date"], y=decomposition.seasonal, title="Seasonality", width=1200, height=300).update_traces(line_color="Orange"), use_container_width=True)
            st.plotly_chart(px.line(x=data_mod["Date"], y=decomposition.resid, title="Residuals", width=1200, height=300).update_traces(line_color="Red", line_dash="dot"), use_container_width=True)
        except Exception as e:
            st.info("Decomposition needs more points; try a longer date range.")

        st.subheader("Model Parameters")
        p = st.slider("Select p", 0, 5, 2)
        d = st.slider("Select d", 0, 5, 1)
        q = st.slider("Select q", 0, 5, 2)
        seasonal_P = st.number_input("Select seasonal P", 0, 24, 12)

        # Safety before modeling (like GitHub)
        data_fit = data_mod.dropna().copy()
        if len(data_fit) < 24:
            st.error("❌ Not enough data for SARIMA model. Choose a longer period.")
        else:
            # Fit SARIMAX similar to GitHub (order=(p,d,q), seasonal=(p,d,q,seasonal_P))
            try:
                model = SARIMAX(
                    data_fit[column],
                    order=(p, d, q),
                    seasonal_order=(p, d, q, seasonal_P),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            except Exception as e:
                st.error(f"Model failed: {e}")
                return

            st.subheader("Forecasting the data")
            forecast_period = st.number_input("Enter forecast period (days)", value=10, min_value=1, step=1)

            # Predictions like GitHub (future)
            pred = model.get_prediction(start=len(data_fit), end=len(data_fit) + forecast_period)
            pred_mean = pred.predicted_mean

            # Index predictions as daily dates starting at end_date (as in GitHub)
            pred_mean.index = pd.date_range(start=end_date, periods=len(pred_mean), freq="D")
            predictions = pd.DataFrame({"Date": pred_mean.index, "predicted_mean": pred_mean.values})
            st.write("## Predictions", predictions)
            st.write("## Actual Data", data_fit)
            st.write("---")

            # Actual vs Predicted (future)
            fig = pgo.Figure()
            
            fig.add_trace(pgo.Scatter(x=data_fit["Date"], y=data_fit[column], mode="lines", name="Actual", line=dict(color="green")))
            fig.add_trace(pgo.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode="lines", name="Predicted", line=dict(color="red")))
            fig.update_layout(title="Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", width=1000, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add button to show plots 
            show_plots = False 
            if st.button('Show Separate Plots'):
                if not show_plots:
                    st.write(px.line(x= data_fit["Date"], y= data_fit[column], title='Actual', width=1200, height=600, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
                    st.write(px.line(x= predictions["Date"], y= predictions["predicted_mean"], title='Predictions', width=1200, height=600, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red'))
                    show_plots = True
                else:
                    show_plots = False
                # Add button to hide plots         
                hide_plots = False 
                if st.button('Hide Separate Plots'):
                    if not hide_plots:  
                        hide_plots = True
                    else:
                        hide_plots = False
                        
            # --- Extra: validation slice (avg variance / MSE) to match your “Avg Variance” requirement
            # Use last N points as validation (if available)
            val_window = st.number_input("Validation window (days) for Avg Variance", min_value=5, max_value=90, value=14, step=1)
            if len(data_fit) > val_window + 10:
                valid_actual = data_fit[column].iloc[-val_window:]
                try:
                    in_sample = model.get_prediction(start=len(data_fit) - val_window, end=len(data_fit) - 1).predicted_mean
                    mae = float(np.nanmean(np.abs(valid_actual.values - in_sample.values)))
                    st.metric("Avg variance (MAE) on validation", f"{mae:,.4f}")
                    # Plot validation avp too
                    fig_val = pgo.Figure()
                    fig_val.add_trace(pgo.Scatter(x=data_fit["Date"].iloc[-val_window:], y=valid_actual, mode="lines", name="Actual (val)"))
                    fig_val.add_trace(pgo.Scatter(x=data_fit["Date"].iloc[-val_window:], y=in_sample, mode="lines", name="Predicted (val)"))
                    fig_val.update_layout(title="Actual vs Predicted (validation slice)", width=1000, height=400)
                    st.plotly_chart(fig_val, use_container_width=True)
                except Exception:
                    st.info("Validation prediction not available for this configuration.")
                    
                    
    # ---------- BUY / SELL panel (live, no st.form)
    st.divider()
    st.markdown("### Trade")

    # 1) Get latest trade price from the CURRENT df (already normalized so 'Adj Close' exists)
    try:
        trade_price = float(df["Adj Close"].dropna().iloc[-1])
    except Exception:
        # Fallback to Close or first numeric column if needed
        if "Close" in df.columns:
            trade_price = float(df["Close"].dropna().iloc[-1])
        else:
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            trade_price = float(df[numcols[0]].dropna().iloc[-1])

    col_buy,col_sell = st.columns(2)

    # ------- BUY -------
    ensure_state(st)

    with col_buy:
        st.markdown("#### Buy")

        # how many shares you can afford
        cash = float(st.session_state.get("cash_balance", 0.0))
        max_affordable = int(max((cash - BUY_FEE) // trade_price, 0))

        qty_buy = st.number_input(
            "Quantity",
            min_value=0,
            max_value=max_affordable,
            value=0,
            step=1,
            key=f"qty_buy_{symbol}",
        )

        buy_fee = BUY_FEE if qty_buy > 0 else 0.0
        buy_subtotal = qty_buy * trade_price
        buy_total = buy_subtotal + buy_fee

        st.write(
            f"Price: **€{trade_price:,.2f}** • "
            f"Subtotal: **€{buy_subtotal:,.2f}** • "
            f"Fee: **€{buy_fee:,.2f}** • "
            f"Total: **€{buy_total:,.2f}**"
        )

        if st.button(
            "Confirm Buy",
            disabled=(qty_buy == 0 or buy_total > cash),
            key=f"buy_btn_{symbol}",
        ):
            _buy(symbol, int(qty_buy), float(trade_price), float(BUY_FEE))

            st.session_state.setdefault("latest_stock_prices", {})
            st.session_state["latest_stock_prices"][symbol] = float(trade_price)

            st.session_state["cash"] = float(st.session_state.get("cash_balance", 0.0))

            record_txn(
                st,
                side="buy",
                asset_class="stocks",       # use "funds" in your funds page
                symbol=symbol,
                qty=int(qty_buy),
                price=float(trade_price),
                fee=float(BUY_FEE if qty_buy > 0 else 0.0),
                ts=dt.datetime.now()
        )

            st.session_state.pop(f"qty_buy_{symbol}", None)

            st.success(
                f"Bought {qty_buy} x {label} at €{trade_price:,.2f}. "
                f"Cash: €{st.session_state.cash_balance:,.2f}"
            )
            st.rerun()

    st.divider()
    if st.button("← Back to Investment Options"):
        st.session_state.page = "page4"
        st.rerun()
        
def render_sell_subpage(SELL_FEE: float = 5.0, FUND_SELL_FEE: float = 5.0, title: str = "Sell Assets"):
    st.header(title)

    # Totals
    _recompute_equity_value()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current Amount (Cash)", f"€{float(st.session_state.get('cash_balance', 0.0)) :,.2f}")
    with c2:
        st.metric("Equity (Stocks + Funds)", f"€{float(st.session_state.get('equity_value', 0.0)) :,.2f}")
    with c3:
        st.metric("Realized Profit (Total)", f"€{float(st.session_state.get('profit_value', 0.0)) :,.2f}")

    # Build combined table
    df = _build_sell_table_all()
    if df.empty:
        st.info("You don’t own any assets yet. Buy stocks or funds first.")
        if st.button("← Back to Investment Options"):
                st.session_state.page = "page4"
                st.rerun()
        return

    st.dataframe(
        df.set_index(["Type", "Symbol"]),
        use_container_width=True,
        height=min(420, 40 + 32 * max(1, len(df)))
    )

    # Selection across both asset types
    options = [f"{r.Type} | {r.Symbol}" for _, r in df.iterrows()]
    choice = st.selectbox("Select an asset to sell", options, key="sell_select_asset")
    atype, sym = [s.strip() for s in choice.split("|", 1)]

    # Row for that asset
    row = df[(df["Type"] == atype) & (df["Symbol"] == sym)].iloc[0]
    available = float(row["Quantity"])
    last_price = float(row["Last Price"])
    avg_cost = float(row["Avg Cost"])

    st.caption(f"{atype} position in {sym}: {available:,.4f} @ avg cost €{avg_cost:,.4f}")

    # Quantity input: int for STOCK, float for FUND
    qkey = f"qty_sell_{atype}_{sym}"
    col1, col2 = st.columns([1, 1])
    with col1:
        if atype == "STOCK":
            st.number_input(
                "Quantity to sell",
                min_value=0,
                max_value=int(available),
                value=0,
                step=1,
                key=qkey,
            )
            qty_sell = int(st.session_state.get(qkey, 0))
        else:
            st.number_input(
                "Units to sell",
                min_value=0.0,
                max_value=float(available),
                value=0.0,
                step=0.0001,
                format="%.4f",
                key=qkey,
            )
            qty_sell = float(st.session_state.get(qkey, 0.0))

    with col2:
        trade_price = last_price  # keep market price; allow override later if desired

    # Preview
    fee = SELL_FEE if atype == "STOCK" else FUND_SELL_FEE
    gross = float(qty_sell) * float(trade_price)
    proceeds = gross - float(fee)
    st.write(
        f"Price: **€{trade_price:,.4f}** • "
        f"Gross: **€{gross:,.2f}** • "
        f"Fee: **€{fee:,.2f}** • "
        f"Proceeds: **€{proceeds:,.2f}**"
    )

    # Confirm
    if st.button(
        "Confirm Sell",
        disabled=(qty_sell == 0 or float(qty_sell) > float(available)),
        key=f"sell_btn_{atype}_{sym}",
    ):
        if atype == "STOCK":
            res = _sell(sym, int(qty_sell), float(trade_price), float(SELL_FEE))
        else:
            res = _fund_sell(sym, float(qty_sell), float(trade_price), float(FUND_SELL_FEE))

        if res:
            realized_pl_delta, final_proceeds = res

            # 🟢 accumulate realized P/L for the Summary page
            st.session_state["realized_pl"] = float(st.session_state.get("realized_pl", 0.0)) + float(realized_pl_delta)

            # 1) Cache latest price for summary valuation
            if atype == "STOCK":
                st.session_state.setdefault("latest_stock_prices", {})
                st.session_state["latest_stock_prices"][sym] = float(trade_price)
            else:
                st.session_state.setdefault("latest_fund_prices", {})
                st.session_state["latest_fund_prices"][sym] = float(trade_price)

            # 2) Mirror your app's cash key to the helpers' default "cash"
            st.session_state["cash"] = float(st.session_state.get("cash_balance", 0.0))

            # 3) Record the transaction + snapshot portfolio history
            record_txn(
                st,
                side="sell",
                asset_class="stocks" if atype == "STOCK" else "funds",
                symbol=sym,
                qty=int(qty_sell) if atype == "STOCK" else float(qty_sell),
                price=float(trade_price),
                fee=float(SELL_FEE if atype == "STOCK" else FUND_SELL_FEE),
                ts=dt.datetime.now()  # or use a user-selected date if you add one later
            )

            st.success(
                f"Sold {qty_sell} x {sym} at €{trade_price:,.4f} • "
                f"Proceeds: €{final_proceeds:,.2f} • Realized P/L: €{realized_pl_delta:,.2f}"
            )
            # reset qty field for this asset
            st.session_state.pop(qkey, None)

        # Refresh table & totals
        st.session_state.sell_view_cache = None
        _recompute_equity_value()
        st.rerun()

    st.divider()
    if st.button("← Back to Investment Options"):
        st.session_state.page = "page4"
        st.rerun()

def render_fund_subpage():
    st.header("Investment Fund Details")

    # Load fake data
    FUNDS_DATA = fund_data()  
    fund_names = list(FUNDS_DATA.keys())
    if not fund_names:
        st.error("No fund data available.")
        return
    fund_name = st.selectbox("Select an Investment Fund", fund_names, key="fund_select")
    fund_df = FUNDS_DATA[fund_name].copy()

    # Current amount
    st.metric("Current Cash Balance (€)", f"{st.session_state.get('cash_balance', 0):,.2f}")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", fund_df["Date"].min())
    with col2:
        end_date = st.date_input("End Date", fund_df["Date"].max())

    mask = (fund_df["Date"] >= pd.to_datetime(start_date)) & (fund_df["Date"] <= pd.to_datetime(end_date))
    filtered = fund_df.loc[mask]

    # Time series chart
    st.subheader(f"Performance of {fund_name}")
    fig, ax = plt.subplots(figsize=(10, 2))  # width, height in inches
    ax.plot(filtered["Date"], filtered["Price"])

    ax.set_xlabel("Date")
    ax.set_ylabel("Fund Value (€)")
    st.pyplot(fig)
    # Fund metrics (Morning Star, Volatility, Expenses)
    metrics = filtered.iloc[-1]
    star_count = int(metrics["Morning_Star"])
    st.markdown("### Fund Metrics")

    stars = "⭐" * star_count + "☆" * (5 - star_count)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Morning Star Rating", stars)
    with col2:
        st.metric("Volatility", f"{metrics['Volatility']:.2f}%")
    with col3:
        st.metric("Operating Expenses", f"{metrics['Expenses']:.2f}%")


    # Buy
    st.divider()
    st.markdown("#### Buy this Fund")

        # --- Get latest price row ---
    # Use filtered date range if available, otherwise whole dataset
    if not filtered.empty:
        latest = filtered.iloc[-1]
    else:
        latest = fund_df.iloc[-1]

    current_price = float(latest["Price"])

    # Show current holding if any
    fpos = st.session_state.fund_holdings.get(fund_name)
    if fpos and fpos.get("qty", 0) > 0:
        st.caption(
            f"Current holding: {float(fpos['qty']):,.4f} units @ avg cost €{float(fpos['avg_cost']):,.4f}"
        )
    # Inputs
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        invest_amount = st.number_input("Amount to invest (€)", min_value=0.0, value=0.0, step=50.0, key=f"fund_in_amt_{fund_name}")
    with c2:
        buy_date = st.date_input("Date of investment", value=pd.Timestamp.today().date(), key=f"fund_buy_date_{fund_name}")
    with c3:
        FUND_BUY_FEE = 5.0  # hardcoded per BRD/prototype
        st.text("")  # spacing
        st.caption(f"Fee per order: €{FUND_BUY_FEE:,.2f}")

    # --- Derived calculations ---
    est_units = ((invest_amount - FUND_BUY_FEE) / current_price) if invest_amount > FUND_BUY_FEE else 0.0
    new_balance = st.session_state.cash_balance - invest_amount if invest_amount > 0 else st.session_state.cash_balance

    st.write(
        f"Price: **€{current_price:,.2f}** • "
        f"Estimated Units: **{est_units:,.4f}** • "
        f"Cash After: **€{new_balance:,.2f}**"
    )

    # --- Confirm Buy button ---
    if st.button(
        f"Confirm Buy ({fund_name})",
        disabled=(invest_amount <= FUND_BUY_FEE),
        key=f"fund_buy_btn_{fund_name}"
    ):
        units = _fund_buy(
            fund_name=fund_name,
            invest_amount=float(invest_amount),
            price=current_price,
            fee=float(FUND_BUY_FEE),
            when=buy_date,
        )
        if units:
        # 1) Cache latest price for summary valuation
            st.session_state.setdefault("latest_fund_prices", {})
            st.session_state["latest_fund_prices"][fund_name] = float(current_price)

            # 2) Mirror your app's cash key to the helpers' default "cash"
            st.session_state["cash"] = float(st.session_state.get("cash_balance", 0.0))

            # 3) Build a timestamp from the chosen buy date (00:00 local)
            ts = dt.datetime.combine(buy_date, dt.datetime.min.time())

            # 4) Record the transaction + snapshot portfolio history
            record_txn(
                st,
                side="buy",
                asset_class="funds",
                symbol=fund_name,
                qty=float(units),                 # units purchased
                price=float(current_price),       # execution price (NAV)
                fee=float(FUND_BUY_FEE),
                ts=ts
            )

            st.success(
                f"Bought ~{units:,.4f} units of {fund_name} at €{current_price:,.2f} "
                f"(Invested €{invest_amount:,.2f}, Fee €{FUND_BUY_FEE:,.2f})."
            )
            st.rerun()
            
    st.divider()
    if st.button("← Back to Investment Options"):
            st.session_state.page = "page4"
            st.rerun()

def render_summary_page():
    ensure_state(st)

    pname = st.session_state.portfolio_name or "Unnamed Portfolio"
    st.markdown(f"## {pname}")

    holdings_df = compute_holdings_df(st)
    totals = compute_totals(st, holdings_df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Amount (Cash)", f"€{totals['cash']:,.2f}")
    k2.metric("Portfolio Value", f"€{totals['portfolio_value']:,.2f}")
    k3.metric("Realized Profit", f"€{totals['realized_pl']:,.2f}")
    k4.metric("Total Profit", f"€{totals['total_profit']:,.2f}")

    st.markdown("### Your Holdings (Stocks + Funds)")
    st.dataframe(
        holdings_df.sort_values(["Class","Asset"]) if not holdings_df.empty else holdings_df,
        use_container_width=True
    )

    st.markdown("### Portfolio Over Time")
    history = st.session_state.get("portfolio_history", [])
    if history:
        hist_df = pd.DataFrame(history).sort_values("ts").drop_duplicates("ts", keep="last")
        base = alt.Chart(hist_df).encode(x=alt.X("ts:T", title="Time"))
        line_total = base.mark_line().encode(y=alt.Y("total_value:Q", title="Total Value (€)"))
        line_portf = base.mark_line(strokeDash=[4,3]).encode(y="portfolio_value:Q")
        line_cash  = base.mark_line(strokeDash=[1,2]).encode(y="cash:Q")
        st.altair_chart(alt.layer(line_total, line_portf, line_cash), use_container_width=True)
    else:
        st.info("No history yet — buy or sell something to populate the chart.")

    with st.expander("Detailed P&L breakdown"):
        cA, cB = st.columns(2)
        cA.write("**Unrealized P/L**")
        cA.write(f"€{totals['unrealized_pl']:,.2f}")
        cB.write("**Realized P/L**")
        cB.write(f"€{totals['realized_pl']:,.2f}")

    st.markdown("### Recent Transactions")
    txns = st.session_state.get("transactions", [])
    if txns:
        tx_df = (pd.DataFrame(txns)
                 .sort_values("ts", ascending=False)
                 .assign(ts=lambda d: pd.to_datetime(d["ts"]).dt.strftime("%Y-%m-%d %H:%M"))
                 [["ts","side","asset_class","symbol","qty","price","fee","cash_after"]]
                 .rename(columns={
                    "ts":"Time","side":"Side","asset_class":"Class","symbol":"Asset",
                    "qty":"Qty","price":"Price","fee":"Fee","cash_after":"Cash After"
                 }))
        st.dataframe(tx_df, use_container_width=True, height=260)
    else:
        st.caption("No transactions yet.")

    st.divider()
    if st.button("← Back to Investment Options"):
            st.session_state.page = "page4"
            st.rerun()
# -------------------------
# Router
# -------------------------
page = st.session_state.page
if page == "page1":
    render_page1()
elif page == "page2":
    render_page2()
elif page == "page3":
    render_page3()
elif page == "page4":
    render_page4()
elif st.session_state.page == "stock_subpage":
    render_stock_subpage1()
elif st.session_state.page == "fund_subpage":
    render_fund_subpage()
elif st.session_state.page == "sell_subpage":
    render_sell_subpage()
elif st.session_state.page == "summary_page":
    render_summary_page()
else:
    render_page1()
