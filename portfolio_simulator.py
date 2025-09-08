# multi_portfolio_lab.py
# ------------------------------------------------------------
# CAPNOW — Multi-Portfolio Simulation Lab (Simplified UI + Per-Deal Actions)
# One main workspace: create deals, jump in calendar, renew/default per deal inline.
# Calendar engine: Mon–Fri collections. Waterfall: 60/40 until 60% ROI, then 25/75; +1.5% fee (on profits).
# Finalization section plus a cross-portfolio comparison tab.
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import date, timedelta

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

# Default investor roster (auto-loaded on new portfolio; editable later)
DEFAULT_INVESTORS = [
    {"name": "Jacobo", "email": "", "commit": 10_000.0},
    {"name": "Albert", "email": "", "commit": 15_000.0},
    {"name": "Yosh",   "email": "", "commit":  5_000.0},
    {"name": "Mike",   "email": "", "commit": 20_000.0},
    {"name": "Juli",   "email": "", "commit": 25_000.0},
    {"name": "Joe",    "email": "", "commit": 10_000.0},
    {"name": "Netty",  "email": "", "commit": 15_000.0},
]

# ------------------------------------------------------------
# Engine
# ------------------------------------------------------------

def launch_portfolio(p: Portfolio):
    if p.launched:
        return
    total = sum(i.commit for i in p.investors)
    if total < p.target_capital:
        st.warning("Total commitments are below target; cannot launch.")
        return
    p.cash = p.target_capital
    p.launched = True
    p.current_date = p.start_date


def add_investor(p: Portfolio, name: str, email: str, commit: float):
    if commit < MIN_TICKET:
        st.warning("Min ticket is $5,000.")
        return
    p.investors.append(Investor(name, email, float(commit)))


def add_deal(p: Portfolio, label: str, amount: float, factor: float, term_days: int, start_date_: date):
    if not p.launched:
        st.warning("Launch portfolio first.")
        return
    if amount > p.cash:
        st.warning("Not enough cash to deploy this deal.")
        return
    end_date = add_biz_days(start_date_, term_days)
    deal = Deal(
        id=ss.next_deal_id,
        label=label or f"Deal {ss.next_deal_id}",
        amount=float(amount),
        factor=float(factor),
        term_days=int(term_days),
        start_date=start_date_,
        end_date=end_date,
    )
    p.deals.append(deal)
    ss.next_deal_id += 1
    p.cash -= amount


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5


def advance_to_date(p: Portfolio, target: date):
    if target < p.current_date:
        p.current_date = target
        return
    d = p.current_date
    while d < target:
        d = d + timedelta(days=1)
        if not _is_weekday(d):
            continue
        for deal in p.deals:
            if deal.completed or deal.defaulted:
                continue
            if deal.start_date < d <= deal.end_date:
                to_add = min(deal.daily, deal.gross - deal.collected)
                deal.collected += to_add
                p.cash += to_add
            if d >= deal.end_date and not deal.completed and not deal.defaulted:
                profit = deal.profit_full
                early = EARLY_SKIM_RATE * profit
                p.cash -= early
                p.early_skim_accum += early
                deal.completed = True
        p.current_date = d


def mark_default(p: Portfolio, deal_id: int, on_date: Optional[date] = None):
    deal = next((x for x in p.deals if x.id == deal_id), None)
    if not deal or deal.completed:
        return
    if on_date and on_date > p.current_date:
        advance_to_date(p, on_date)
    deal.defaulted = True
    deal.completed = True


def renew_deal(p: Portfolio, deal_id: int, new_amount: float, new_factor: float, new_term_days: int, on_date: Optional[date] = None):
    deal = next((x for x in p.deals if x.id == deal_id), None)
    if not deal or deal.completed or deal.defaulted:
        return
    renewal_date = on_date or p.current_date
    if renewal_date > p.current_date:
        advance_to_date(p, renewal_date)
    remaining = max(0.0, deal.gross - deal.collected)
    profit = deal.profit_full
    early = EARLY_SKIM_RATE * profit
    p.cash -= early
    p.early_skim_accum += early
    deal.completed = True
    if new_amount < remaining:
        st.warning("New advance must cover payoff of the old remaining balance.")
        return
    fresh_cash = new_amount - remaining
    if fresh_cash > p.cash:
        st.warning("Not enough cash for the fresh portion of renewal.")
        return
    p.cash -= fresh_cash
    new_end = add_biz_days(renewal_date, new_term_days)
    new_deal = Deal(
        id=ss.next_deal_id,
        label=f"Renewal of {deal.label}",
        amount=float(new_amount),
        factor=float(new_factor),
        term_days=int(new_term_days),
        start_date=renewal_date,
        end_date=new_end,
        renewed_from=deal.id,
    )
    ss.next_deal_id += 1
    p.deals.append(new_deal)

# ------------------------------------------------------------
# Metrics + Waterfall
# ------------------------------------------------------------

def cap_table(p: Portfolio) -> pd.DataFrame:
    if not p.investors:
        return pd.DataFrame(columns=["Name","Email","Commit","% Ownership"])
    total = p.target_capital
    rows = [{
        "Name": i.name,
        "Email": i.email,
        "Commit": float(i.commit),
        "% Ownership": (float(i.commit)/total) if total else 0.0,
    } for i in p.investors]
    return pd.DataFrame(rows)


def realized_profit(p: Portfolio) -> float:
    prof = 0.0
    for d in p.deals:
        principal_repaid = min(d.collected, d.amount)
        prof += max(0.0, d.collected - principal_repaid)
    return prof


def outstanding_principal(p: Portfolio) -> float:
    out = 0.0
    for d in p.deals:
        principal_repaid = min(d.collected, d.amount)
        out += max(0.0, d.amount - principal_repaid)
    return out


def to_be_collected(p: Portfolio) -> float:
    rem = 0.0
    for d in p.deals:
        if d.defaulted:
            continue
        rem += max(0.0, d.gross - d.collected)
    return rem


def renewal_and_default_rates(p: Portfolio):
    total = len(p.deals)
    if total == 0:
        return (0.0, 0.0, 0.0, 0.0)
    renewed = sum(1 for d in p.deals if d.renewed_from is not None)
    defaulted = sum(1 for d in p.deals if d.defaulted)
    total_amt = sum(d.amount for d in p.deals)
    renewed_amt = sum(d.amount for d in p.deals if d.renewed_from is not None)
    default_amt = sum(d.amount for d in p.deals if d.defaulted)
    return (
        renewed / total,
        renewed_amt / total_amt if total_amt else 0.0,
        defaulted / total,
        default_amt / total_amt if total_amt else 0.0,
    )


def year_end_waterfall(p: Portfolio) -> Dict[str, float]:
    principal = p.target_capital
    hurdle_amt = HURDLE_ROI * principal
    ending_cash = p.cash
    cash_profits_after_skims = realized_profit(p)

    # Phase 1: 60/40 until investors hit 60% ROI
    gross_needed_for_hurdle = hurdle_amt / PRE_HURDLE_SPLIT_INV if PRE_HURDLE_SPLIT_INV > 0 else float("inf")
    used_in_phase1 = min(cash_profits_after_skims, gross_needed_for_hurdle)
    inv_from_phase1 = used_in_phase1 * PRE_HURDLE_SPLIT_INV
    cap_from_phase1 = used_in_phase1 * PRE_HURDLE_SPLIT_CAP

    # Phase 2: 25/75 on remainder
    remainder_pool = max(0.0, cash_profits_after_skims - used_in_phase1)
    inv_from_phase2 = remainder_pool * POST_HURDLE_SPLIT_INV
    cap_from_phase2 = remainder_pool * POST_HURDLE_SPLIT_CAP

    # Mgmt fee = 1.5% of PROFITS (not ending cash)
    mgmt_fee = MGMT_FEE * cash_profits_after_skims

    investor_total = principal + inv_from_phase1 + inv_from_phase2 - mgmt_fee
    capnow_total = p.early_skim_accum + cap_from_phase1 + cap_from_phase2 + mgmt_fee

    return {
        "principal": principal,
        "hurdle": hurdle_amt,
        "phase1_inv": inv_from_phase1,
        "phase1_cap": cap_from_phase1,
        "phase2_inv": inv_from_phase2,
        "phase2_cap": cap_from_phase2,
        "mgmt_fee": mgmt_fee,
        "investor_total": investor_total,
        "capnow_total": capnow_total,
        "ending_cash": ending_cash,
        "early_skims": p.early_skim_accum,
        "cash_profit_after_skims": cash_profits_after_skims,
    }

# ------------------------------------------------------------
# Sidebar — New portfolio + Investors & Launch
# ------------------------------------------------------------
st.sidebar.header("🧪 Portfolio Lab")
with st.sidebar.expander("New Portfolio", expanded=True):
    name = st.text_input("Name", value="Base 2025")
    c1, c2 = st.columns(2)
    with c1:
        start_dt = st.date_input("Start", value=date(2025,1,1), key="p_start")
    with c2:
        end_dt = st.date_input("End", value=date(2026,1,1), key="p_end")
    target = st.number_input("Target Capital $", min_value=0.0, value=100_000.0, step=1000.0, key="p_target")
    if st.button("Create", use_container_width=True):
        pid = ss.next_portfolio_id
        ss.next_portfolio_id += 1
        newp = Portfolio(id=pid, name=name, start_date=start_dt, end_date=end_dt, target_capital=target)
        newp.investors = [Investor(i["name"], i["email"], float(i["commit"])) for i in DEFAULT_INVESTORS]
        ss.portfolios[pid] = newp
        ss.selected_pid = pid

if ss.portfolios:
    options = {f"#{pid} — {ss.portfolios[pid].name}": pid for pid in sorted(ss.portfolios)}
    pick = st.sidebar.selectbox("Open", list(options.keys()), index=len(options)-1)
    ss.selected_pid = options[pick]

if ss.selected_pid is None:
    st.info("Create or select a portfolio in the left sidebar.")
    st.stop()

p: Portfolio = ss.portfolios[ss.selected_pid]

with st.sidebar.expander("Investors & Launch", expanded=True):
    inv_df = pd.DataFrame([{"name": i.name, "email": i.email, "commit": i.commit} for i in p.investors])
    edited = st.data_editor(
        inv_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "email": st.column_config.TextColumn("Email"),
            "commit": st.column_config.NumberColumn("Commit $", step=1000.0, min_value=0.0, format="$%0.0f"),
        },
        key=f"inv_editor_{p.id}"
    )
    p.investors = [Investor(str(r.get("name","")), str(r.get("email","")), float(r.get("commit") or 0.0)) for r in edited.to_dict(orient="records")]

    st.markdown("— or quick add —")
    n1, n2 = st.columns(2)
    with n1:
        inv_name = st.text_input("Name", key=f"inv_name_{p.id}")
    with n2:
        inv_email = st.text_input("Email", key=f"inv_email_{p.id}")
    inv_amt = st.number_input("Commit $ (≥5k)", min_value=0.0, step=1000.0, key=f"inv_amt_{p.id}")
    if st.button("Add investor", key=f"add_inv_{p.id}"):
        add_investor(p, inv_name, inv_email, inv_amt)

    # Cap table preview
    df_cap = cap_table(p)
    if not df_cap.empty:
        show = df_cap.copy()
        show["Commit"] = show["Commit"].map(dollars)
        show["% Ownership"] = (show["% Ownership"]*100).map(lambda v: f"{v:.2f}%")
        st.dataframe(show, use_container_width=True, height=220)

    if st.button("🚀 Launch", disabled=p.launched or sum(i.commit for i in p.investors) < p.target_capital):
        launch_portfolio(p)
    st.caption(f"Cash: {dollars(p.cash)} · Early skims: {dollars(p.early_skim_accum)} · Date: {p.current_date}")

# ------------------------------------------------------------
# MAIN — Single Workspace (Deals + Calendar + Actions + KPIs)
# ------------------------------------------------------------

st.markdown(f"## Portfolio #{p.id}: {p.name}")

# Top row: Date controls + quick moves
c0, c1, c2, c3 = st.columns([2,2,2,2])
with c0:
    st.write("**Sim date**:", p.current_date)
with c1:
    jump_date = st.date_input("Go to date", value=p.current_date, key=f"jump_{p.id}")
with c2:
    if st.button("Advance to date", key=f"adv_{p.id}"):
        advance_to_date(p, jump_date)
with c3:
    colx, coly = st.columns(2)
    with colx:
        if st.button("+10 weekdays", key=f"q10_{p.id}"):
            advance_to_date(p, add_biz_days(p.current_date, 10))
    with coly:
        if st.button("To End", key=f"qend_{p.id}"):
            advance_to_date(p, p.end_date)

st.markdown("---")

# Deal creator row (simple)
cc1, cc2, cc3, cc4, cc5 = st.columns([2,2,2,2,2])
with cc1:
    d_label = st.text_input("Label", value=f"Deal {ss.next_deal_id}", key=f"dlab_{p.id}")
with cc2:
    d_amount = st.number_input("Amount $", min_value=0.0, value=10_000.0, step=1000.0, key=f"damt_{p.id}")
with cc3:
    d_factor = st.number_input("Factor", min_value=1.0, value=1.40, step=0.01, format="%.2f", key=f"dfac_{p.id}")
with cc4:
    d_term = st.number_input("Term (bdays)", min_value=1, value=90, step=1, key=f"dterm_{p.id}")
with cc5:
    d_start = st.date_input("Start date", value=p.current_date, key=f"dstart_{p.id}")

if st.button("Deploy Deal", key=f"deploy_{p.id}"):
    add_deal(p, d_label, d_amount, d_factor, d_term, d_start)

# Live KPIs
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Cash", dollars(p.cash))
with k2: st.metric("Realized Profit", dollars(realized_profit(p)))
with k3: st.metric("Outstanding Principal", dollars(outstanding_principal(p)))
with k4: st.metric("To-Be-Collected", dollars(to_be_collected(p)))
with k5: st.metric("Early Skims", dollars(p.early_skim_accum))

rr_cnt, rr_amt, df_cnt, df_amt = renewal_and_default_rates(p)
st.caption(
    f"Renewal rate: {rr_cnt*100:.1f}% (count) · {rr_amt*100:.1f}% ($-weighted) · "
    f"Default rate: {df_cnt*100:.1f}% (count) · {df_amt*100:.1f}% ($-weighted)"
)

# Deal table
if p.deals:
    df = pd.DataFrame([{
        "id": d.id,
        "label": d.label,
        "amount": d.amount,
        "factor": d.factor,
        "term_days": d.term_days,
        "start_date": d.start_date,
        "end_date": d.end_date,
        "daily": d.daily,
        "gross": d.gross,
        "collected": d.collected,
        # NEW: progress metrics vs current sim date
        "days_left": max(0, d.term_days - biz_days_between(d.start_date, min(p.current_date, d.end_date))),
        "progress": f"{min(d.term_days, biz_days_between(d.start_date, min(p.current_date, d.end_date)))}/{d.term_days}",
        "completed": d.completed,
        "defaulted": d.defaulted,
        "renewed_from": d.renewed_from,
    } for d in p.deals])
    show = df.copy()
    for c in ["amount","daily","gross","collected"]:
        show[c] = show[c].map(dollars)
    st.dataframe(show, use_container_width=True, height=360)

    # Inline per-deal actions
    st.markdown("### Quick Actions per Deal")
    for d in p.deals:
        with st.container(border=True):
            a1, a2, a3, a4, a5, a6 = st.columns([3,2,2,2,1.5,1.5])
            with a1:
                elapsed = min(d.term_days, biz_days_between(d.start_date, min(p.current_date, d.end_date)))
                st.markdown(f"**#{d.id} — {d.label}**  ")
                st.caption(
                    f"Start: {d.start_date} · End: {d.end_date} · "
                    f"Collected: {dollars(d.collected)} / {dollars(d.gross)} · "
                    f"Days: {elapsed}/{d.term_days}"
                )
            with a2:
                act_date = st.date_input("Action date", value=p.current_date, key=f"ad_{p.id}_{d.id}")
            with a3:
                if st.button("Default", key=f"defbtn_{p.id}_{d.id}"):
                    mark_default(p, d.id, act_date)
                    st.toast(f"Deal #{d.id} defaulted on {act_date}")
            with a4:
                rn_amount = st.number_input(
                    "Renew: new advance $", min_value=0.0,
                    value=float(max(d.amount, d.gross - d.collected)),
                    step=1000.0, key=f"rnamt_{p.id}_{d.id}"
                )
            with a5:
                rn_factor = st.number_input("Factor", min_value=1.0, value=float(d.factor), step=0.01, format="%.2f", key=f"rnfac_{p.id}_{d.id}")
            with a6:
                rn_term = st.number_input("Term (bdays)", min_value=1, value=int(d.term_days), step=1, key=f"rnterm_{p.id}_{d.id}")
            b1, b2 = st.columns([1,1])
            with b1:
                if st.button("Apply Renewal", key=f"rnbtn_{p.id}_{d.id}"):
                    renew_deal(p, d.id, rn_amount, rn_factor, rn_term, act_date)
                    st.toast(f"Deal #{d.id} renewed on {act_date}")
            with b2:
                st.write("")

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
