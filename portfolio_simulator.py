# multi_portfolio_lab.py
# ------------------------------------------------------------
# CAPNOW — Multi-Portfolio Simulation Lab (Simplified UI + Per-Deal Actions)
# One main workspace: create deals, jump in calendar, renew/default per deal inline.
# Calendar engine: Mon–Fri collections. Waterfall: 60/40 until 60% ROI, then 25/75; +1.5% fee (on profits).
# Finalization section at bottom with investor payouts and save-runs scaffold.
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# App config
# ------------------------------------------------------------
st.set_page_config(page_title="CAPNOW – Portfolio Lab", page_icon="🧪", layout="wide")

# ------------------------------------------------------------
# Business-day helpers (Mon–Fri)
# ------------------------------------------------------------
WEEKMASK = "Mon Tue Wed Thu Fri"
BD = pd.offsets.CustomBusinessDay(weekmask=WEEKMASK)


def add_biz_days(start: date, n: int) -> date:
    return (pd.Timestamp(start) + n * BD).date()


def biz_days_between(d1: date, d2: date) -> int:
    """Business days strictly after d1 up to and including d2 (Mon–Fri)."""
    if d2 <= d1:
        return 0
    rng = pd.date_range(start=d1, end=d2, freq=BD)
    # exclude start so elapsed=0 on start day
    return max(0, len(rng) - 1)


def dollars(x: float) -> str:
    try:
        return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"
    except Exception:
        return "$0.00"

# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------
EARLY_SKIM_RATE = 0.20
HURDLE_ROI = 0.60  # on original principal
PRE_HURDLE_SPLIT_INV = 0.60
PRE_HURDLE_SPLIT_CAP = 0.40
POST_HURDLE_SPLIT_INV = 0.25
POST_HURDLE_SPLIT_CAP = 0.75
MGMT_FEE = 0.015            # applied to PROFITS only
MIN_TICKET = 5_000.0

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class Investor:
    name: str
    email: str
    commit: float

@dataclass
class Deal:
    id: int
    label: str
    amount: float
    factor: float
    term_days: int
    start_date: date
    end_date: date
    collected: float = 0.0
    completed: bool = False
    defaulted: bool = False
    renewed_from: Optional[int] = None

    @property
    def gross(self) -> float:
        return self.amount * self.factor

    @property
    def daily(self) -> float:
        return self.gross / max(1, self.term_days)

    @property
    def profit_full(self) -> float:
        return max(0.0, self.gross - self.amount)

@dataclass
class Portfolio:
    id: int
    name: str
    start_date: date
    end_date: date
    target_capital: float
    launched: bool = False
    cash: float = 0.0
    early_skim_accum: float = 0.0
    current_date: date = None
    investors: List[Investor] = None
    deals: List[Deal] = None

    def __post_init__(self):
        if self.current_date is None:
            self.current_date = self.start_date
        if self.investors is None:
            self.investors = []
        if self.deals is None:
            self.deals = []

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
ss = st.session_state
if "portfolios" not in ss:
    ss.portfolios: Dict[int, Portfolio] = {}
if "next_portfolio_id" not in ss:
    ss.next_portfolio_id = 1
if "next_deal_id" not in ss:
    ss.next_deal_id = 1
if "selected_pid" not in ss:
    ss.selected_pid = None
if "saved_runs" not in ss:
    ss.saved_runs = {}

# ------------------------------------------------------------
# Engine
# ------------------------------------------------------------

# ... [ENGINE FUNCTIONS UNCHANGED] ...

# ------------------------------------------------------------
# Metrics + Waterfall
# ------------------------------------------------------------

# ... [METRICS FUNCTIONS UNCHANGED] ...

# ------------------------------------------------------------
# Sidebar — Portfolio + Investors (simple)
# ------------------------------------------------------------
# ... [SIDEBAR CODE UNCHANGED] ...

# ------------------------------------------------------------
# MAIN — Single Workspace (Deals + Calendar + Actions + KPIs)
# ------------------------------------------------------------
# ... [MAIN CODE UNCHANGED] ...

# ------------------------------------------------------------
# FINALIZATION — Run/Finalize Year + Cross-Portfolio Comparison
# ------------------------------------------------------------

# helper to snapshot current portfolio into comparison list
def _snapshot_for_compare(p: Portfolio) -> Dict[str, object]:
    wf_local = year_end_waterfall(p)
    rr_cnt, rr_amt, df_cnt, df_amt = renewal_and_default_rates(p)
    elapsed_bd = biz_days_between(p.start_date, p.current_date)
    investor_roi = (wf_local["investor_total"] / p.target_capital - 1.0) if p.target_capital else 0.0
    return {
        "portfolio_id": p.id,
        "name": p.name,
        "start": p.start_date,
        "current": p.current_date,
        "elapsed_bdays": elapsed_bd,
        "principal": float(p.target_capital),
        "realized_profit": float(wf_local["cash_profit_after_skims"]),
        "early_skims": float(wf_local["early_skims"]),
        "mgmt_fee": float(wf_local["mgmt_fee"]),
        "investor_total": float(wf_local["investor_total"]),
        "capnow_total": float(wf_local["capnow_total"]),
        "investor_roi": float(investor_roi),
        "renewal_rate_cnt": float(rr_cnt),
        "default_rate_cnt": float(df_cnt),
        "deals": len(p.deals),
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

# Tabs: finalize (this) | compare (all)
st.markdown("---")
st.markdown("## 📌 Finalize & Compare")

final_tabs = st.tabs(["✅ Run / Finalize (this portfolio)", "📊 All Portfolio Runs (comparison)"])

# --- Tab 1: This portfolio finalize ---
with final_tabs[0]:
    wf = year_end_waterfall(p)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Realized Profit (YTD)", dollars(wf["cash_profit_after_skims"]))
    with c2: st.metric("CapNow Early Skims (sum)", dollars(wf["early_skims"]))
    with c3: st.metric("CapNow Final Component", dollars(wf["phase1_cap"] + wf["phase2_cap"]))
    with c4: st.metric("Fees Collected (1.5% of profits)", dollars(wf["mgmt_fee"]))

    c5, c6 = st.columns(2)
    with c5: st.metric("Investors' Total Distribution", dollars(wf["investor_total"]))
    with c6: st.metric("CapNow – All-in (incl. fees)", dollars(wf["capnow_total"]))

    # Detailed payout table per investor
    ct = cap_table(p)
    if not ct.empty:
        pay = ct.copy()
        pay["Payout"] = pay["% Ownership"] * wf["investor_total"]
        pay["ROI %"] = ((pay["Payout"] - pay["Commit"]) / pay["Commit"]) * 100.0
        show = pay.copy()
        show["Commit"] = show["Commit"].map(dollars)
        show["% Ownership"] = (show["% Ownership"]*100).map(lambda v: f"{v:.2f}%")
        show["Payout"] = show["Payout"].map(dollars)
        show["ROI %"] = show["ROI %"].map(lambda v: f"{v:.2f}%")
        st.dataframe(show, use_container_width=True)
    else:
        st.info("No investors yet.")

    if st.button("📌 Save snapshot to comparison", use_container_width=True):
        snap = _snapshot_for_compare(p)
        ss.saved_runs.setdefault("runs", []).append(snap)
        st.success("Saved! Check the 'All Portfolio Runs' tab.")

# --- Tab 2: Comparison table ---
with final_tabs[1]:
    runs = ss.saved_runs.get("runs", [])
    if not runs:
        st.info("No saved runs yet. Use the button in the first tab to add a snapshot.")
    else:
        df_runs = pd.DataFrame(runs)
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ["investor_roi desc", "elapsed_bdays asc", "realized_profit desc", "capnow_total desc"],
            index=0
        )
        ascending = sort_by.endswith("asc")
        col = sort_by.split()[0]
        df_runs = df_runs.sort_values(col, ascending=ascending)

        pretty = df_runs.copy()
        pretty["principal"] = pretty["principal"].map(dollars)
        pretty["investor_total"] = pretty["investor_total"].map(dollars)
        pretty["capnow_total"] = pretty["capnow_total"].map(dollars)
        pretty["realized_profit"] = pretty["realized_profit"].map(dollars)
        pretty["early_skims"] = pretty["early_skims"].map(dollars)
        pretty["mgmt_fee"] = pretty["mgmt_fee"].map(dollars)
        pretty["investor_roi"] = (pretty["investor_roi"]*100).map(lambda v: f"{v:.2f}%")
        pretty["renewal_rate_cnt"] = (pretty["renewal_rate_cnt"]*100).map(lambda v: f"{v:.1f}%")
        pretty["default_rate_cnt"] = (pretty["default_rate_cnt"]*100).map(lambda v: f"{v:.1f}%")
        # Column order
        cols = [
            "timestamp","portfolio_id","name","elapsed_bdays","deals",
            "principal","investor_total","investor_roi","capnow_total",
            "realized_profit","early_skims","mgmt_fee",
            "renewal_rate_cnt","default_rate_cnt","start","current",
        ]
        pretty = pretty[cols]
        st.dataframe(pretty, use_container_width=True)
