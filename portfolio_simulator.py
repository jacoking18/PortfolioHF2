# multi_portfolio_lab.py
# ------------------------------------------------------------
# CAPNOW — Multi-Portfolio Simulation Lab (Simplified UI)
# One primary workspace focused on: creating deals + moving in calendar year.
# Calendar engine = Mon–Fri collections. Renew/Default on any date.
# Waterfall = 60/40 until investors hit 60% ROI, then 25/75; +1.5% fee on final cash.
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
MGMT_FEE = 0.015
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
    # Always append (no upsert)
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
    # Determine gross profits consumed to pay investors' 60% via 60/40 split
    gross_needed_for_hurdle = hurdle_amt / PRE_HURDLE_SPLIT_INV if PRE_HURDLE_SPLIT_INV > 0 else float("inf")
    used_in_phase1 = min(cash_profits_after_skims, gross_needed_for_hurdle)
    inv_from_phase1 = used_in_phase1 * PRE_HURDLE_SPLIT_INV
    cap_from_phase1 = used_in_phase1 * PRE_HURDLE_SPLIT_CAP

    # Phase 2: 25/75 on remainder
    remainder_pool = max(0.0, cash_profits_after_skims - used_in_phase1)
    inv_from_phase2 = remainder_pool * POST_HURDLE_SPLIT_INV
    cap_from_phase2 = remainder_pool * POST_HURDLE_SPLIT_CAP

    mgmt_fee = MGMT_FEE * ending_cash

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
# Sidebar — Portfolio + Investors (simple)
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
        ss.portfolios[pid] = Portfolio(id=pid, name=name, start_date=start_dt, end_date=end_dt, target_capital=target)
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
    n1, n2 = st.columns(2)
    with n1:
        inv_name = st.text_input("Name", key=f"inv_name_{p.id}")
    with n2:
        inv_email = st.text_input("Email", key=f"inv_email_{p.id}")
    inv_amt = st.number_input("Commit $ (≥5k)", min_value=0.0, step=1000.0, key=f"inv_amt_{p.id}")
    if st.button("Add investor", key=f"add_inv_{p.id}"):
        add_investor(p, inv_name, inv_email, inv_amt)
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
        "completed": d.completed,
        "defaulted": d.defaulted,
        "renewed_from": d.renewed_from,
    } for d in p.deals])
    show = df.copy()
    for c in ["amount","daily","gross","collected"]:
        show[c] = show[c].map(dollars)
    st.dataframe(show, use_container_width=True, height=360)

    # Action Center — quick checkmark style (select a deal, pick date, renew/default)
    st.markdown("**Action Center**")
    ac1, ac2, ac3, ac4 = st.columns([2,2,4,4])
    with ac1:
        deal_id_sel = st.number_input("Deal ID", min_value=1, step=1, value=int(df.id.iloc[-1]))
    with ac2:
        action_date = st.date_input("Action date", value=p.current_date, key=f"actdate_{p.id}")
    with ac3:
        do_default = st.checkbox("Default on date", value=False, key=f"def_{p.id}")
        if do_default and st.button("Apply Default", key=f"apply_def_{p.id}"):
            mark_default(p, int(deal_id_sel), action_date)
            st.success("Default applied.")
    with ac4:
        do_renew = st.checkbox("Renew on date", value=False, key=f"ren_{p.id}")
        rn_amount = st.number_input("New advance $", min_value=0.0, step=1000.0, key=f"rnamt_{p.id}")
        rn_factor = st.number_input("New factor", min_value=1.0, value=1.40, step=0.01, format="%.2f", key=f"rnfac_{p.id}")
        rn_term = st.number_input("New term (bdays)", min_value=1, value=90, step=1, key=f"rnterm_{p.id}")
        if do_renew and st.button("Apply Renewal", key=f"apply_rn_{p.id}"):
            renew_deal(p, int(deal_id_sel), rn_amount, rn_factor, rn_term, action_date)
            st.success("Renewal applied.")

# Waterfall preview (always visible)
st.markdown("---")
st.markdown("### Year-End Waterfall (Preview)")
wf = year_end_waterfall(p)
w1, w2, w3, w4 = st.columns(4)
with w1: st.metric("Principal", dollars(wf["principal"]))
with w2: st.metric("Hurdle (60%)", dollars(wf["hurdle"]))
with w3: st.metric("Mgmt Fee (1.5%)", dollars(wf["mgmt_fee"]))
with w4: st.metric("Ending Cash", dollars(wf["ending_cash"]))
w5, w6, w7, w8 = st.columns(4)
with w5: st.metric("Phase1 → Investors (60%)", dollars(wf["phase1_inv"]))
with w6: st.metric("Phase1 → CAPNOW (40%)", dollars(wf["phase1_cap"]))
with w7: st.metric("Phase2 → Investors (25%)", dollars(wf["phase2_inv"]))
with w8: st.metric("Phase2 → CAPNOW (75%)", dollars(wf["phase2_cap"]))
wx1, wx2 = st.columns(2)
with wx1: st.metric("Investors — Total", dollars(wf["investor_total"]))
with wx2: st.metric("CAPNOW — Total", dollars(wf["capnow_total"]))
