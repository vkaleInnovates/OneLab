"""
Payments Reconciliation Dashboard — Streamlit Application.

An advanced, interactive dashboard for month-end reconciliation
between platform transactions and bank settlements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generation import generate_datasets, print_summary
from reconciliation import reconcile, print_report


# ═══════════════════════════════════════════════════════════════════════
#  Page Configuration
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Payments Reconciliation System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    .metric-card .icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
    .metric-card .value {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #e2e8f0, #f8fafc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        font-size: 0.82rem; color: #94a3b8;
        font-weight: 500; text-transform: uppercase;
        letter-spacing: 0.5px; margin-top: 2px;
    }

    /* Accent colors via card variants */
    .metric-blue::before { background: linear-gradient(90deg, #6366f1, #818cf8); }
    .metric-green::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .metric-red::before { background: linear-gradient(90deg, #ef4444, #f87171); }
    .metric-amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .metric-purple::before { background: linear-gradient(90deg, #a855f7, #c084fc); }
    .metric-teal::before { background: linear-gradient(90deg, #14b8a6, #2dd4bf); }
    .metric-orange::before { background: linear-gradient(90deg, #f97316, #fb923c); }
    .metric-rose::before { background: linear-gradient(90deg, #f43f5e, #fb7185); }

    /* ── Gap cards ── */
    .gap-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
        cursor: pointer;
    }
    .gap-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .gap-card .gap-title {
        font-size: 1rem; font-weight: 700; color: #f1f5f9;
        margin-bottom: 0.4rem;
    }
    .gap-card .gap-count {
        font-size: 1.6rem; font-weight: 800; color: #e2e8f0;
    }
    .gap-card .gap-amount {
        font-size: 0.85rem; color: #94a3b8;
    }
    .gap-card .gap-desc {
        font-size: 0.82rem; color: #64748b;
        margin-top: 0.5rem; line-height: 1.5;
    }

    .gap-high { border-left: 4px solid #ef4444; }
    .gap-medium { border-left: 4px solid #f59e0b; }
    .gap-low { border-left: 4px solid #10b981; }

    /* ── Explanation cards ── */
    .explanation-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .explanation-card .exp-type {
        font-weight: 700; font-size: 0.9rem;
    }
    .explanation-card .exp-text {
        color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;
        margin-top: 0.5rem;
    }
    .severity-high { color: #f87171; }
    .severity-medium { color: #fbbf24; }
    .severity-low { color: #34d399; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #000000 !important;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    /* ── Assumptions / Limitations lists ── */
    .info-list {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.08);
    }
    .info-list li {
        color: #cbd5e1;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }

    /* ── Toast (test results) ── */
    .test-pass { color: #34d399; }
    .test-fail { color: #f87171; }

    /* ── Hide Streamlit defaults ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        position: relative;
        overflow: hidden;
    }
    .header-banner::after {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    }
    .header-banner h1 {
        font-size: 1.8rem; font-weight: 800; color: #f1f5f9;
        margin: 0 0 0.3rem 0;
    }
    .header-banner p {
        color: #94a3b8; font-size: 0.95rem; margin: 0;
    }

    /* table row colors */
    .row-matched { background-color: rgba(16,185,129,0.08) !important; }
    .row-gap { background-color: rgba(239,68,68,0.08) !important; }
    .row-partial { background-color: rgba(245,158,11,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Data Loading (cached)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(seed: int = 42):
    datasets = generate_datasets(seed=seed)
    results = reconcile(
        datasets["platform_df"],
        datasets["bank_df"],
        datasets["metadata"]["reconciliation_period"],
    )
    return datasets, results


# ═══════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        seed = st.number_input(
            "Random Seed", min_value=1, max_value=9999, value=42,
            help="Change to generate a different dataset"
        )

        if st.button("🔄 Regenerate Data", use_container_width=True, type="primary"):
            st.cache_data.clear()

        st.markdown("---")
        st.markdown("### 🔍 Filters")

        gap_types = ["All", "Missing Settlement", "Late Settlement",
                     "Duplicate", "Orphan Refund", "Rounding Difference"]
        selected_gap = st.selectbox("Gap Type", gap_types)

        date_range = st.date_input(
            "Date Range",
            value=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-05")),
            min_value=pd.Timestamp("2026-01-01"),
            max_value=pd.Timestamp("2026-02-28"),
        )

        amount_range = st.slider(
            "Amount Range ($)", -1000.0, 6000.0, (-1000.0, 6000.0),
            step=50.0
        )

        st.markdown("---")
        st.markdown("### 📊 Dashboard")
        st.markdown("Built with **Streamlit** + **Plotly**")
        st.markdown("Data: Synthetic (January 2026)")

    return seed, selected_gap, date_range, amount_range


# ═══════════════════════════════════════════════════════════════════════
#  Render Functions
# ═══════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="header-banner">
        <h1>🏦 Payments Reconciliation System</h1>
        <p>Month-end reconciliation • January 2026 • Platform vs Bank Settlements</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_metrics(summary: dict):
    cols = st.columns(4)

    kpis = [
        ("📋", summary["total_platform_transactions"], "Total Transactions", "metric-blue"),
        ("✅", summary["matched_count"], "Matched", "metric-green"),
        ("❌", summary["gap_count"] + summary["partial_match_count"],
         "Unmatched / Issues", "metric-red"),
        ("💰", f"${summary['net_difference']:,.4f}", "Net Difference",
         "metric-rose" if abs(summary["net_difference"]) > 1 else "metric-teal"),
    ]

    for col, (icon, value, label, css) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card {css}">
                <div class="icon">{icon}</div>
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Second row of KPIs
    st.markdown("")
    cols2 = st.columns(4)
    kpis2 = [
        ("🔴", summary["missing_in_bank"], "Missing Settlements", "metric-red"),
        ("⏰", summary["late_settlements"], "Late Settlements", "metric-amber"),
        ("🔁", summary["duplicate_transactions"], "Duplicates", "metric-orange"),
        ("⚠️", summary["orphan_refunds"], "Orphan Refunds", "metric-purple"),
    ]
    for col, (icon, value, label, css) in zip(cols2, kpis2):
        with col:
            st.markdown(f"""
            <div class="metric-card {css}">
                <div class="icon">{icon}</div>
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_totals_bar(summary: dict):
    cols = st.columns(3)
    with cols[0]:
        st.metric("Platform Total", f"${summary['platform_total_amount']:,.2f}")
    with cols[1]:
        st.metric("Bank Total", f"${summary['bank_total_amount']:,.2f}")
    with cols[2]:
        diff = summary['net_difference']
        st.metric("Net Difference", f"${diff:,.4f}",
                  delta=f"${diff:,.4f}",
                  delta_color="inverse" if diff != 0 else "off")


def render_gap_analysis(results: dict):
    st.markdown('<div class="section-header">🔎 Gap Analysis</div>',
                unsafe_allow_html=True)

    gap_data = [
        {
            "title": "🔴 Missing Settlements",
            "count": results["summary"]["missing_in_bank"],
            "severity": "high",
            "amount": results["missing_in_bank"]["amount_platform"].sum()
            if "amount_platform" in results["missing_in_bank"].columns
            else (results["missing_in_bank"]["amount"].sum()
                  if "amount" in results["missing_in_bank"].columns else 0),
            "desc": "Platform transactions with no corresponding bank settlement",
        },
        {
            "title": "⏰ Late Settlements",
            "count": results["summary"]["late_settlements"],
            "severity": "medium",
            "amount": results["late_settlements"]["amount_platform"].sum()
            if not results["late_settlements"].empty else 0,
            "desc": "Transactions settled in the next month (February)",
        },
        {
            "title": "🔁 Duplicates",
            "count": results["summary"]["duplicate_transactions"],
            "severity": "high",
            "amount": results["duplicates"]["amount"].sum()
            if not results["duplicates"].empty else 0,
            "desc": "Duplicate transaction_ids in platform dataset",
        },
        {
            "title": "⚠️ Orphan Refunds",
            "count": results["summary"]["orphan_refunds"],
            "severity": "high",
            "amount": abs(results["orphan_refunds"]["amount"].sum())
            if not results["orphan_refunds"].empty else 0,
            "desc": "Refunds without a matching original transaction",
        },
        {
            "title": "🔢 Rounding Differences",
            "count": results["summary"]["rounding_differences"],
            "severity": "low",
            "amount": abs(results["rounding_diffs"]["amount_diff"].sum())
            if not results["rounding_diffs"].empty else 0,
            "desc": "Sub-cent discrepancies from floating-point processing",
        },
    ]

    cols = st.columns(len(gap_data))
    for col, gd in zip(cols, gap_data):
        with col:
            st.markdown(f"""
            <div class="gap-card gap-{gd['severity']}">
                <div class="gap-title">{gd['title']}</div>
                <div class="gap-count">{gd['count']}</div>
                <div class="gap-amount">Impact: ${gd['amount']:,.2f}</div>
                <div class="gap-desc">{gd['desc']}</div>
            </div>
            """, unsafe_allow_html=True)


def render_charts(results: dict, datasets: dict):
    st.markdown('<div class="section-header">📈 Visualizations</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📊 Gap Amount by Type",
        "🥧 Matched vs Unmatched",
        "📅 Daily Volume"
    ])

    with tab1:
        render_gap_amount_chart(results)

    with tab2:
        render_match_pie(results)

    with tab3:
        render_daily_volume(datasets)


def render_gap_amount_chart(results: dict):
    s = results["summary"]
    gap_types = ["Missing", "Late", "Duplicates", "Orphan Refunds", "Rounding"]

    # Calculate amounts for each type
    missing_amt = (results["missing_in_bank"]["amount_platform"].sum()
                   if "amount_platform" in results["missing_in_bank"].columns
                   else results["missing_in_bank"]["amount"].sum()
                   if "amount" in results["missing_in_bank"].columns and len(results["missing_in_bank"]) > 0
                   else 0)
    late_amt = (results["late_settlements"]["amount_platform"].sum()
                if not results["late_settlements"].empty else 0)
    dup_amt = (results["duplicates"]["amount"].sum()
               if not results["duplicates"].empty else 0)
    orphan_amt = (abs(results["orphan_refunds"]["amount"].sum())
                  if not results["orphan_refunds"].empty else 0)
    rounding_amt = (abs(results["rounding_diffs"]["amount_diff"].sum())
                    if not results["rounding_diffs"].empty else 0)

    amounts = [missing_amt, late_amt, dup_amt, orphan_amt, rounding_amt]

    colors = ['#ef4444', '#f59e0b', '#f97316', '#a855f7', '#14b8a6']

    fig = go.Figure(go.Bar(
        x=gap_types,
        y=amounts,
        marker_color=colors,
        marker_line_width=0,
        text=[f"${a:,.2f}" for a in amounts],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=13, family='Inter'),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        title=dict(text="Gap Impact by Type", font=dict(size=18)),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)",
                   title="Amount ($)"),
        height=420,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_match_pie(results: dict):
    s = results["summary"]
    labels = ["Matched", "Partial Match", "Gap"]
    values = [s["matched_count"], s["partial_match_count"], s["gap_count"]]
    colors = ['#10b981', '#f59e0b', '#ef4444']

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color='#0f172a', width=2)),
        hole=0.45,
        textfont=dict(size=14, color='#f1f5f9'),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Pct: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        title=dict(text="Reconciliation Status", font=dict(size=18)),
        height=420,
        margin=dict(t=60, b=40),
        legend=dict(font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_daily_volume(datasets: dict):
    platform_df = datasets["platform_df"].copy()
    bank_df = datasets["bank_df"].copy()

    # Daily counts
    plat_daily = (platform_df.groupby(platform_df["transaction_date"].dt.date)
                  .size().reset_index(name="count"))
    plat_daily.columns = ["date", "count"]
    plat_daily["source"] = "Platform"

    bank_daily = (bank_df.groupby(bank_df["settlement_date"].dt.date)
                  .size().reset_index(name="count"))
    bank_daily.columns = ["date", "count"]
    bank_daily["source"] = "Bank"

    combined = pd.concat([plat_daily, bank_daily])

    fig = px.line(combined, x="date", y="count", color="source",
                  color_discrete_map={"Platform": "#6366f1", "Bank": "#14b8a6"},
                  markers=True)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        title=dict(text="Daily Transaction vs Settlement Volume", font=dict(size=18)),
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title="Count", gridcolor="rgba(148,163,184,0.08)"),
        height=420, margin=dict(t=60, b=40),
        legend_title_text="",
    )
    fig.update_traces(line_width=2.5)
    st.plotly_chart(fig, use_container_width=True)


def render_data_tables(datasets: dict, results: dict,
                       selected_gap: str, date_range, amount_range):
    st.markdown('<div class="section-header">📑 Data Tables</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📋 Platform Transactions",
        "🏦 Bank Settlements",
        "🔀 Reconciliation Results"
    ])

    with tab1:
        render_platform_table(datasets, date_range, amount_range)
    with tab2:
        render_bank_table(datasets, date_range, amount_range)
    with tab3:
        render_recon_table(results, selected_gap, date_range, amount_range)


def render_platform_table(datasets: dict, date_range, amount_range):
    df = datasets["platform_df"].copy()

    # Apply filters
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df["transaction_date"] >= start) & (df["transaction_date"] <= end)]
    df = df[(df["amount"] >= amount_range[0]) & (df["amount"] <= amount_range[1])]

    st.markdown(f"**{len(df)} transactions** shown")

    display_df = df.copy()
    display_df["transaction_date"] = display_df["transaction_date"].dt.strftime("%Y-%m-%d")
    display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_df, use_container_width=True, height=400)


def render_bank_table(datasets: dict, date_range, amount_range):
    df = datasets["bank_df"].copy()

    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df["settlement_date"] >= start) & (df["settlement_date"] <= end)]
    df = df[(df["amount"] >= amount_range[0]) & (df["amount"] <= amount_range[1])]

    st.markdown(f"**{len(df)} settlements** shown")

    display_df = df.copy()
    display_df["settlement_date"] = display_df["settlement_date"].dt.strftime("%Y-%m-%d")
    display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.4f}")
    st.dataframe(display_df, use_container_width=True, height=400)


def render_recon_table(results: dict, selected_gap: str,
                       date_range, amount_range):
    df = results["classified"].copy()

    # Filter by gap type
    if selected_gap != "All":
        gap_type_map = {
            "Missing Settlement": lambda d: d[d["_merge"] == "left_only"],
            "Late Settlement": lambda d: d[d["transaction_id"].isin(
                results["late_settlements"]["transaction_id"].tolist()
                if not results["late_settlements"].empty else []
            )],
            "Duplicate": lambda d: d[d["transaction_id"].isin(
                results["duplicates"]["transaction_id"].tolist()
                if not results["duplicates"].empty else []
            )],
            "Orphan Refund": lambda d: d[d["transaction_id"].isin(
                results["orphan_refunds"]["transaction_id"].tolist()
                if not results["orphan_refunds"].empty else []
            )],
            "Rounding Difference": lambda d: d[d["transaction_id"].isin(
                results["rounding_diffs"]["transaction_id"].tolist()
                if not results["rounding_diffs"].empty else []
            )],
        }
        if selected_gap in gap_type_map:
            df = gap_type_map[selected_gap](df)

    # Apply amount filter on platform amount
    if "amount_platform" in df.columns:
        df = df[
            (df["amount_platform"].fillna(0) >= amount_range[0]) &
            (df["amount_platform"].fillna(0) <= amount_range[1])
        ]

    st.markdown(f"**{len(df)} records** shown")

    # Show selected columns
    display_cols = ["transaction_id", "classification"]
    if "transaction_date" in df.columns:
        display_cols.append("transaction_date")
    if "amount_platform" in df.columns:
        display_cols.append("amount_platform")
    if "settlement_date" in df.columns:
        display_cols.append("settlement_date")
    if "amount_bank" in df.columns:
        display_cols.append("amount_bank")

    display_df = df[display_cols].copy()

    # Color rows by classification
    def highlight_rows(row):
        if row["classification"] == "Gap":
            return ['background-color: rgba(239,68,68,0.1)'] * len(row)
        elif row["classification"] == "Partial Match":
            return ['background-color: rgba(245,158,11,0.1)'] * len(row)
        else:
            return ['background-color: rgba(16,185,129,0.06)'] * len(row)

    styled = display_df.style.apply(highlight_rows, axis=1)
    st.dataframe(styled, use_container_width=True, height=400)


def render_drill_down(results: dict):
    st.markdown('<div class="section-header">🔬 Drill-Down Explorer</div>',
                unsafe_allow_html=True)

    gap_choice = st.selectbox(
        "Select gap type to drill into:",
        ["Missing Settlements", "Late Settlements", "Duplicates",
         "Orphan Refunds", "Rounding Differences"],
        key="drill_down_select"
    )

    drill_map = {
        "Missing Settlements": results["missing_in_bank"],
        "Late Settlements": results["late_settlements"],
        "Duplicates": results["duplicates"],
        "Orphan Refunds": results["orphan_refunds"],
        "Rounding Differences": results["rounding_diffs"],
    }

    detail_df = drill_map.get(gap_choice, pd.DataFrame())

    if detail_df.empty:
        st.info(f"No {gap_choice.lower()} detected.")
    else:
        st.markdown(f"**{len(detail_df)} record(s) found**")
        st.dataframe(detail_df, use_container_width=True, height=350)

        # Show related explanations
        related_exps = [
            e for e in results["explanations"]
            if e["type"].lower().replace(" ", "") in gap_choice.lower().replace(" ", "")
            or gap_choice.lower().startswith(e["type"].lower().split()[0])
        ]
        if related_exps:
            st.markdown("**Explanations:**")
            for exp in related_exps[:5]:
                severity_class = f"severity-{exp['severity'].lower()}"
                st.markdown(f"""
                <div class="explanation-card">
                    <div class="exp-type">
                        <span class="{severity_class}">[{exp['severity']}]</span>
                        {exp['transaction_id']}
                    </div>
                    <div class="exp-text">{exp['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)


def render_explanations(results: dict):
    st.markdown('<div class="section-header">💡 Explanation Engine</div>',
                unsafe_allow_html=True)

    explanations = results["explanations"]

    if not explanations:
        st.success("No gaps detected — perfect reconciliation!")
        return

    # Group by type
    exp_types = {}
    for exp in explanations:
        exp_types.setdefault(exp["type"], []).append(exp)

    for exp_type, exps in exp_types.items():
        with st.expander(f"**{exp_type}** — {len(exps)} issue(s)", expanded=False):
            for exp in exps:
                severity_class = f"severity-{exp['severity'].lower()}"
                st.markdown(f"""
                <div class="explanation-card">
                    <div class="exp-type">
                        <span class="{severity_class}">[{exp['severity']}]</span>
                        {exp['transaction_id']}
                    </div>
                    <div class="exp-text">{exp['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)


def render_test_results():
    st.markdown('<div class="section-header">🧪 Test Results</div>',
                unsafe_allow_html=True)

    from tests import run_all_tests
    import io, sys

    # Capture test output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    passed, failed = run_all_tests()
    output = buffer.getvalue()
    sys.stdout = old_stdout

    cols = st.columns(3)
    with cols[0]:
        st.metric("Passed", passed)
    with cols[1]:
        st.metric("Failed", failed)
    with cols[2]:
        st.metric("Total", passed + failed)

    # Progress bar
    total = passed + failed
    if total > 0:
        st.progress(passed / total, text=f"{passed}/{total} tests passing")

    with st.expander("View test output", expanded=False):
        st.code(output, language="text")


def render_assumptions():
    st.markdown('<div class="section-header">📌 Assumptions</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-list">
    <ul>
        <li><strong>Settlement Delay:</strong> Bank settles transactions 1–2 business days after the platform records them.</li>
        <li><strong>Currency:</strong> All transactions are in USD. No multi-currency or FX conversions.</li>
        <li><strong>Matching Key:</strong> Transactions are matched using exact <code>transaction_id</code>. No fuzzy matching.</li>
        <li><strong>No Partial Settlements:</strong> Each transaction is settled fully in one batch — no split/partial settlements.</li>
        <li><strong>Refund Identification:</strong> Refunds are identified by negative amounts and <code>type='refund'</code>.</li>
        <li><strong>Month Boundary:</strong> Reconciliation is for January 2026. Transactions from prior/subsequent months are excluded from the primary analysis.</li>
        <li><strong>Rounding Threshold:</strong> Discrepancies under $0.01 are classified as rounding differences, not mismatches.</li>
        <li><strong>Synthetic Data:</strong> All data is generated synthetically with a fixed seed for reproducibility.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def render_limitations():
    st.markdown('<div class="section-header">⚠️ Real-World Limitations</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-list">
    <ul>
        <li><strong>Multi-Currency / FX:</strong> The system assumes a single currency (USD). In production, exchange rate fluctuations between transaction and settlement dates would introduce additional discrepancies not handled here.</li>
        <li><strong>Partial / Split Settlements:</strong> Banks may split a single transaction across multiple settlement batches (e.g., due to risk holds or reserve requirements). This system assumes 1:1 transaction-to-settlement mapping.</li>
        <li><strong>Chargebacks & Disputes:</strong> The model does not account for chargebacks, which can reverse a settled transaction weeks or months later, creating reconciliation gaps outside the standard window.</li>
        <li><strong>Timezone Handling:</strong> All dates are treated as naive (no timezone). In reality, a transaction at 11:59 PM EST might appear on a different date in the bank's UTC-based system.</li>
        <li><strong>Batch Aggregation:</strong> Some banks settle in aggregated batches (one lump sum per day), requiring a different decomposition strategy not implemented here.</li>
        <li><strong>Holiday / Weekend Delays:</strong> The system does not account for bank holidays or weekends, which can extend settlement windows beyond 2 days.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Main App
# ═══════════════════════════════════════════════════════════════════════

def main():
    seed, selected_gap, date_range, amount_range = render_sidebar()

    datasets, results = load_data(seed)

    render_header()
    render_kpi_metrics(results["summary"])
    st.markdown("")
    render_totals_bar(results["summary"])
    st.markdown("")
    render_gap_analysis(results)
    st.markdown("")
    render_charts(results, datasets)
    st.markdown("")
    render_data_tables(datasets, results, selected_gap, date_range, amount_range)
    st.markdown("")
    render_drill_down(results)
    st.markdown("")
    render_explanations(results)
    st.markdown("")
    render_test_results()
    st.markdown("")
    render_assumptions()
    st.markdown("")
    render_limitations()


if __name__ == "__main__":
    main()
