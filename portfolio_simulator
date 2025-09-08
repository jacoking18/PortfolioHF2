# multi_portfolio_lab.py
# ------------------------------------------------------------
# CAPNOW — Multi-Portfolio Simulation Lab (MVP)
# Calendar year engine (Mon–Fri collections), renewals, defaults,
# waterfall with 60/40 until investors hit 60% ROI, then 25/75.
# 1.5% mgmt fee on final pool (cash-only for now).
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import date, datetime, timedelta

import math
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# UI / App config
# ------------------------------------------------------------
st.set_page_config(page_title="CAPNOW – Multi-Portfolio Lab (MVP)", page_icon="🧪", layout="wide")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
WEEKMASK = "Mon Tue Wed Thu Fri"
BD = pd.offsets.CustomBusinessDay(weekmask=WEEKMASK)


def to_biz_days(start: date, end: date) -> int:
    if end < start:
        return 0
    # include start as day 0; count steps between
    rng = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=BD)
    return max(0, len(rng) - 1)


def add_biz_days(start: date, n: int) -> date:
    return (pd.Timestamp(start) + n * BD).date()


def dollars(x: float) -> str:
    try:
        return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"
    except Exception:
        return "$0.00"

# ------------------------------------------------------------
# Global settings (can be per-portfolio later)
# ------------------------------------------------------------
EARLY_SKIM_RATE = 0.20  # CAPNOW skim on deal profit at completion
HURDLE_ROI = 0.60       # 60% on original principal
PRE_HURDLE_SPLIT_INV = 0.60
PRE_HURDLE_SPLIT_CAP = 0.40
POST_HURDLE_SPLIT_INV = 0.25
POST_HURDLE_SPLIT_CAP = 0.75
MGMT_FEE = 0.015        # on final pool size (cash-only)
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
    term_days: int  # business days (Mon–Fri)
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
if "portfolios" not in st.session_state:
    st.session_state["portfolios"]: Dict[int, Portfolio] = {}
if "next_portfolio_id" not in st.session_state:
    st.session_state["next_portfolio_id"] = 1
if "next_deal_id" not in st.session_state:
    st.session_state["next_deal_id"] = 1
if "selected_pid" not in st.session_state:
    st.session_state["selected_pid"] = None
if "saved_runs" not in st.session_state:
    st.session_state["saved_runs"]: Dict[str, Dict] = {}

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
    # upsert by email
    existing = next((x for x in p.investors if x.email == email), None)
    if existing:
        existing.name, existing.commit = name, commit
    else:
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
        id=st.session_state["next_deal_id"],
        label=label or f"Deal {st.session_state['next_deal_id']}",
        amount=float(amount),
        factor=float(factor),
        term_days=int(term_days),
        start_date=start_date_,
        end_date=end_date,
    )
    p.deals.append(deal)
    st.session_state["next_deal_id"] += 1
    p.cash -= amount


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5  # Mon=0 .. Sun=6


def advance_to_date(p: Portfolio, target: date):
    if target < p.current_date:
        # pointer-only move backward (no rollback)
        p.current_date = target
        return
    d = p.current_date
    while d < target:
        d = d + timedelta(days=1)
        if not _is_weekday(d):
            continue
        # daily collections for open deals
        for deal in p.deals:
            if deal.completed or deal.defaulted:
                continue
            if deal.start_date < d <= deal.end_date:
                to_add = min(deal.daily, deal.gross - deal.collected)
                deal.collected += to_add
                p.cash += to_add
            if d >= deal.end_date and not deal.completed and not deal.defaulted:
                # maturity → skim once
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
    # advance up to on_date first to accrue until that day
    if on_date and on_date > p.current_date:
        advance_to_date(p, on_date)
    deal.defaulted = True
    deal.completed = True  # stops future collections


def renew_deal(p: Portfolio, deal_id: int, new_amount: float, new_factor: float, new_term_days: int, on_date: Optional[date] = None):
    """Renewal counts as maturity of the old deal on renewal day.
    New advance = payoff(old remaining balance) + fresh cash to merchant.
    Factor/term apply to the NEW advance."""
    deal = next((x for x in p.deals if x.id == deal_id), None)
    if not deal or deal.completed or deal.defaulted:
        return
    # move time forward if needed
    renewal_date = on_date or p.current_date
    if renewal_date > p.current_date:
        advance_to_date(p, renewal_date)
    # Compute remaining balance to payoff
    remaining = max(0.0, deal.gross - deal.collected)
    # Treat old deal as matured now → take skim
    profit = deal.profit_full
    early = EARLY_SKIM_RATE * profit
    p.cash -= early
    p.early_skim_accum += early
    deal.completed = True
    # New deal deployment: payoff + fresh → new_amount must be >= remaining
    if new_amount < remaining:
        st.warning("New advance must cover payoff of the old remaining balance.")
        return
    fresh_cash = new_amount - remaining
    # We receive the payoff (remaining) back immediately (it flows to cash via new advance),
    # but we then deploy the full new_amount. Net cash change = -fresh_cash.
    if fresh_cash > p.cash:
        st.warning("Not enough cash for the fresh portion of renewal.")
        return
    p.cash -= fresh_cash
    new_end = add_biz_days(renewal_date, new_term_days)
    new_deal = Deal(
        id=st.session_state["next_deal_id"],
        label=f"Renewal of {deal.label}",
        amount=float(new_amount),
        factor=float(new_factor),
        term_days=int(new_term_days),
        start_date=renewal_date,
        end_date=new_end,
        renewed_from=deal.id,
    )
    st.session_state["next_deal_id"] += 1
    p.deals.append(new_deal)

# ------------------------------------------------------------
# Metrics & Waterfall
# ------------------------------------------------------------

def cap_table(p: Portfolio) -> pd.DataFrame:
    if not p.investors:
        return pd.DataFrame(columns=["Name","Email","Commit","% Ownership"])
    total = p.target_capital
    rows = []
    for i in p.investors:
        rows.append({
            "Name": i.name,
            "Email": i.email,
            "Commit": float(i.commit),
            "% Ownership": float(i.commit) / total if total else 0.0,
        })
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
    """Implements Jacobo's rule:
    - 60/40 split UNTIL investors reach 60% ROI on principal.
    - Then 25/75 on the remainder.
    - 1.5% fee on final pool (cash-only now).
    NOTE: Uses realized (cash) profit and early skims already deducted from cash.
    """
    principal = p.target_capital
    hurdle_amt = HURDLE_ROI * principal  # 60% on principal

    # Total cash result at end (cash-only definition for fee & distributions)
    ending_cash = p.cash

    # Compute total profits AFTER skims already removed from cash.
    # realized_profit() is profit that has arrived in cash (before skim at completion)
    # Early skims were directly deducted from cash and stored in p.early_skim_accum.
    # We need the portion that investors/CAPNOW will share now (post-skim cash profits):
    cash_profits_after_skims = realized_profit(p)

    # Phase 1: split 60/40 until investors reach hurdle
    inv_from_phase1 = min(hurdle_amt, cash_profits_after_skims) * PRE_HURDLE_SPLIT_INV
    cap_from_phase1 = min(hurdle_amt, cash_profits_after_skims) * PRE_HURDLE_SPLIT_CAP

    # If profits were less than hurdle, phase1 pays proportionally (not reaching hurdle fully)
    remaining_needed_for_hurdle = hurdle_amt - inv_from_phase1

    # Phase 2: if hurdle reached, split remainder 25/75
    remainder_pool = max(0.0, cash_profits_after_skims - (inv_from_phase1 / PRE_HURDLE_SPLIT_INV)) if PRE_HURDLE_SPLIT_INV > 0 else 0.0
    # Above formula converts investor amount back to gross phase1 profits consumed.

    inv_from_phase2 = 0.0
    cap_from_phase2 = 0.0
    if remaining_needed_for_hurdle <= 1e-9:  # hurdle achieved
        inv_from_phase2 = remainder_pool * POST_HURDLE_SPLIT_INV
        cap_from_phase2 = remainder_pool * POST_HURDLE_SPLIT_CAP

    # Mgmt fee on final pool (cash-only)
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
# UI — Portfolio Manager
# ------------------------------------------------------------

st.sidebar.header("🧪 Multi-Portfolio Lab")
with st.sidebar.expander("New Portfolio", expanded=True):
    name = st.text_input("Name", value="Base 2025")
    colA, colB = st.columns(2)
    with colA:
        start_dt = st.date_input("Start", value=date(2025,1,1))
    with colB:
        end_dt = st.date_input("End", value=date(2026,1,1))
    target = st.number_input("Target Capital $", min_value=0.0, value=100_000.0, step=1000.0)
    if st.button("Create Portfolio", use_container_width=True):
        pid = st.session_state["next_portfolio_id"]
        st.session_state["next_portfolio_id"] += 1
        st.session_state["portfolios"][pid] = Portfolio(id=pid, name=name, start_date=start_dt, end_date=end_dt, target_capital=target)
        st.session_state["selected_pid"] = pid

# Portfolio list
pids = sorted(st.session_state["portfolios"].keys())
if pids:
    options = {f"#{pid} — {st.session_state['portfolios'][pid].name}": pid for pid in pids}
    choice = st.sidebar.selectbox("Open portfolio", list(options.keys()), index=len(options)-1)
    st.session_state["selected_pid"] = options[choice]

# Body
st.markdown("# CAPNOW — Multi-Portfolio Simulation Lab (MVP)")

if st.session_state["selected_pid"] is None:
    st.info("Create or select a portfolio in the left sidebar to begin.")
    st.stop()

p: Portfolio = st.session_state["portfolios"][st.session_state["selected_pid"]]

# Header bar
col1, col2, col3, col4 = st.columns([3,2,2,2])
with col1:
    st.subheader(f"Portfolio #{p.id}: {p.name}")
with col2:
    st.metric("Start → End", f"{p.start_date} → {p.end_date}")
with col3:
    st.metric("Target", dollars(p.target_capital))
with col4:
    st.metric("Status", "Launched" if p.launched else "Draft")

# Tabs
tabs = st.tabs(["Investors","Deals","Advance Time","Metrics","Waterfall","Compare Runs"])

# Investors
with tabs[0]:
    st.markdown("### Investors")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_name = st.text_input("Name", key=f"inv_name_{p.id}")
    with c2:
        in_email = st.text_input("Email", key=f"inv_email_{p.id}")
    with c3:
        in_commit = st.number_input("Commit $", min_value=0.0, step=1000.0, key=f"inv_amt_{p.id}")
    if st.button("Add/Update Investor", key=f"add_inv_{p.id}"):
        add_investor(p, in_name, in_email, in_commit)
    inv_df = cap_table(p)
    if not inv_df.empty:
        inv_df_show = inv_df.copy()
        inv_df_show["Commit"] = inv_df_show["Commit"].map(dollars)
        inv_df_show["% Ownership"] = (inv_df_show["% Ownership"]*100).map(lambda v: f"{v:.2f}%")
        st.dataframe(inv_df_show, use_container_width=True)
    colx, coly = st.columns([1,3])
    with colx:
        if st.button("🚀 Launch Portfolio", disabled=p.launched or sum(i.commit for i in p.investors) < p.target_capital):
            launch_portfolio(p)
    with coly:
        st.write("Cash:", dollars(p.cash), "· Early skims:", dollars(p.early_skim_accum), "· Current date:", p.current_date)

# Deals
with tabs[1]:
    st.markdown("### Deals")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        d_label = st.text_input("Label", value=f"Deal {st.session_state['next_deal_id']}", key=f"dlab_{p.id}")
    with c2:
        d_amount = st.number_input("Amount $", min_value=0.0, value=10_000.0, step=1000.0, key=f"damt_{p.id}")
    with c3:
        d_factor = st.number_input("Factor", min_value=1.0, value=1.40, step=0.01, format="%.2f", key=f"dfac_{p.id}")
    with c4:
        d_term = st.number_input("Term (bdays)", min_value=1, value=90, step=1, key=f"dterm_{p.id}")
    with c5:
        d_start = st.date_input("Start date", value=p.current_date, key=f"dstart_{p.id}")
    if st.button("Deploy Deal", key=f"deploy_{p.id}"):
        add_deal(p, d_label, d_amount, d_factor, d_term, d_start)

    # Deal grid
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
        st.dataframe(show, use_container_width=True)

        st.markdown("**Actions**")
        colA, colB, colC, colD = st.columns([2,2,2,2])
        with colA:
            sel_id = st.number_input("Deal ID", min_value=1, step=1, value=int(df.id.iloc[-1]))
        with colB:
            if st.button("Default this deal"):
                mark_default(p, int(sel_id))
        with colC:
            rn_amount = st.number_input("Renew: new advance $", min_value=0.0, step=1000.0)
        with colD:
            rn_factor = st.number_input("Renew: new factor", min_value=1.0, value=1.40, step=0.01, format="%.2f")
        r2, r3 = st.columns(2)
        with r2:
            rn_term = st.number_input("Renew: new term (bdays)", min_value=1, value=90, step=1)
        with r3:
            rn_date = st.date_input("Renew on date", value=p.current_date)
        if st.button("Renew now"):
            renew_deal(p, int(sel_id), rn_amount, rn_factor, rn_term, rn_date)

# Advance Time
with tabs[2]:
    st.markdown("### Advance Time (Mon–Fri collections)")
    st.write("Current sim date:", p.current_date)
    jump_date = st.date_input("Go to date", value=p.current_date)
    if st.button("Advance"):
        advance_to_date(p, jump_date)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("+10 weekdays"):
            advance_to_date(p, add_biz_days(p.current_date, 10))
    with col2:
        if st.button("+30 weekdays"):
            advance_to_date(p, add_biz_days(p.current_date, 30))
    with col3:
        if st.button("To portfolio end"):
            advance_to_date(p, p.end_date)

# Metrics
with tabs[3]:
    st.markdown("### Metrics (Live)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Cash", dollars(p.cash))
    with col2:
        st.metric("Realized Profit", dollars(realized_profit(p)))
    with col3:
        st.metric("Outstanding Principal", dollars(outstanding_principal(p)))
    with col4:
        st.metric("To-Be-Collected", dollars(to_be_collected(p)))
    with col5:
        st.metric("Early Skims (CAPNOW)", dollars(p.early_skim_accum))
    rr_cnt, rr_amt, df_cnt, df_amt = renewal_and_default_rates(p)
    st.write(
        f"Renewal rate: {rr_cnt*100:.1f}% (count) · {rr_amt*100:.1f}% ($-weighted) — "
        f"Default rate: {df_cnt*100:.1f}% (count) · {df_amt*100:.1f}% ($-weighted)"
    )

# Waterfall (preview at now; final at end)
with tabs[4]:
    st.markdown("### Year-End Waterfall (Preview)")
    wf = year_end_waterfall(p)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Principal", dollars(wf["principal"]))
    with c2: st.metric("Hurdle (60%)", dollars(wf["hurdle"]))
    with c3: st.metric("Mgmt Fee (1.5%)", dollars(wf["mgmt_fee"]))
    c4, c5, c6, c7 = st.columns(4)
    with c4: st.metric("Phase1 → Investors (60%)", dollars(wf["phase1_inv"]))
    with c5: st.metric("Phase1 → CAPNOW (40%)", dollars(wf["phase1_cap"]))
    with c6: st.metric("Phase2 → Investors (25%)", dollars(wf["phase2_inv"]))
    with c7: st.metric("Phase2 → CAPNOW (75%)", dollars(wf["phase2_cap"]))
    c8, c9 = st.columns(2)
    with c8: st.metric("Investors — Total", dollars(wf["investor_total"]))
    with c9: st.metric("CAPNOW — Total", dollars(wf["capnow_total"]))
    st.caption(f"Ending cash: {dollars(wf['ending_cash'])} · Cash profit after skims: {dollars(wf['cash_profit_after_skims'])} · Early skims sum: {dollars(wf['early_skims'])}")

# Compare Runs
with tabs[5]:
    st.markdown("### Save & Compare Runs")
    run_name = st.text_input("Run name", value=f"Run @ {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Save this run"):
        st.session_state["saved_runs"][run_name] = {
            "portfolio_id": p.id,
            "name": p.name,
            "wf": wf,
            "cash": p.cash,
            "realized_profit": realized_profit(p),
            "renewal_cnt": renewal_and_default_rates(p)[0],
            "default_cnt": renewal_and_default_rates(p)[2],
        }
        st.success("Run saved.")
    if st.session_state["saved_runs"]:
        runs_df = pd.DataFrame([
            {
                "Run": k,
                "Portfolio": v["name"],
                "Investors Total": v["wf"]["investor_total"],
                "CAPNOW Total": v["wf"]["capnow_total"],
                "Realized Profit": v["realized_profit"],
                "Renewal % (count)": v["renewal_cnt"]*100,
                "Default % (count)": v["default_cnt"]*100,
            }
            for k,v in st.session_state["saved_runs"].items()
        ])
        show = runs_df.copy()
        show["Investors Total"] = show["Investors Total"].map(dollars)
        show["CAPNOW Total"] = show["CAPNOW Total"].map(dollars)
        show["Realized Profit"] = show["Realized Profit"].map(dollars)
        st.dataframe(show, use_container_width=True)
