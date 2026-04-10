"""
Reconciliation Engine for Payments Reconciliation System.

Performs outer-join matching, gap detection, classification, and
human-readable explanation generation using pandas.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ── Classification labels ───────────────────────────────────────────────
LABEL_MATCHED = "Matched"
LABEL_GAP = "Gap"
LABEL_PARTIAL = "Partial Match"


def reconcile(platform_df: pd.DataFrame, bank_df: pd.DataFrame,
              recon_month: str = "2026-01") -> dict:
    """
    Run full reconciliation between platform and bank datasets.

    Args:
        platform_df: Platform transactions DataFrame
        bank_df: Bank settlements DataFrame
        recon_month: Month under reconciliation (YYYY-MM)

    Returns:
        dict with reconciliation results
    """
    results = {}

    # ── 1. Detect duplicates BEFORE merging ─────────────────────────────
    duplicates = detect_duplicates(platform_df)
    results["duplicates"] = duplicates

    # Deduplicate platform for matching (keep first occurrence)
    platform_deduped = platform_df.drop_duplicates(
        subset="transaction_id", keep="first"
    ).copy()

    # ── 2. Outer join on transaction_id ─────────────────────────────────
    merged = pd.merge(
        platform_deduped,
        bank_df,
        on="transaction_id",
        how="outer",
        suffixes=("_platform", "_bank"),
        indicator=True,
    )

    # ── 3. Detect missing settlements ───────────────────────────────────
    missing_in_bank = detect_missing_settlements(merged)
    results["missing_in_bank"] = missing_in_bank

    # ── 4. Detect late settlements ──────────────────────────────────────
    late_settlements = detect_late_settlements(merged, recon_month)
    results["late_settlements"] = late_settlements

    # ── 5. Detect orphan refunds ────────────────────────────────────────
    orphan_refunds = detect_orphan_refunds(platform_deduped)
    results["orphan_refunds"] = orphan_refunds

    # ── 6. Detect rounding differences ──────────────────────────────────
    rounding_diffs = detect_rounding_diffs(merged)
    results["rounding_diffs"] = rounding_diffs

    # ── 7. Classify every record ────────────────────────────────────────
    classified = classify_records(merged, late_settlements, rounding_diffs)
    results["classified"] = classified

    # ── 8. Summary statistics ───────────────────────────────────────────
    results["summary"] = build_summary(
        platform_df, bank_df, classified, missing_in_bank,
        late_settlements, duplicates, orphan_refunds, rounding_diffs
    )

    # ── 9. Explanations ────────────────────────────────────────────────
    results["explanations"] = generate_explanations(results)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Gap Detection Functions
# ═══════════════════════════════════════════════════════════════════════

def detect_duplicates(platform_df: pd.DataFrame) -> pd.DataFrame:
    """Find duplicate transaction_ids in the platform dataset."""
    if platform_df.empty or "transaction_id" not in platform_df.columns:
        return pd.DataFrame()
    dup_mask = platform_df.duplicated(subset="transaction_id", keep=False)
    duplicates = platform_df[dup_mask].copy()
    if not duplicates.empty:
        duplicates["gap_type"] = "Duplicate"
    return duplicates


def detect_missing_settlements(merged: pd.DataFrame) -> pd.DataFrame:
    """Find platform transactions with no bank settlement."""
    missing = merged[merged["_merge"] == "left_only"].copy()
    if not missing.empty:
        missing["gap_type"] = "Missing Settlement"
    return missing


def detect_late_settlements(merged: pd.DataFrame,
                            recon_month: str) -> pd.DataFrame:
    """
    Find transactions settled in a different month than the
    transaction month, or with settlement delay > 2 days.
    """
    both = merged[merged["_merge"] == "both"].copy()
    if both.empty:
        return pd.DataFrame()

    both["transaction_date"] = pd.to_datetime(both["transaction_date"])
    both["settlement_date"] = pd.to_datetime(both["settlement_date"])

    both["txn_month"] = both["transaction_date"].dt.to_period("M")
    both["stl_month"] = both["settlement_date"].dt.to_period("M")
    both["delay_days"] = (
        both["settlement_date"] - both["transaction_date"]
    ).dt.days

    recon_period = pd.Period(recon_month, freq="M")

    late_mask = (
        (both["txn_month"] == recon_period) &
        (both["stl_month"] != recon_period)
    ) | (both["delay_days"] > 2)

    late = both[late_mask].copy()
    if not late.empty:
        late["gap_type"] = "Late Settlement"
    return late


def detect_orphan_refunds(platform_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find refund transactions (negative amounts) whose original
    transaction does not exist in the platform dataset.

    Uses heuristic: a refund is orphaned if no charge with the
    same absolute amount exists from the same customer.
    """
    refunds = platform_df[platform_df["amount"] < 0].copy()
    charges = platform_df[platform_df["amount"] > 0]

    if refunds.empty:
        return pd.DataFrame()

    orphans = []
    for _, refund in refunds.iterrows():
        abs_amount = abs(refund["amount"])
        # Look for a matching charge from the same customer
        match = charges[
            (charges["customer_name"] == refund["customer_name"]) &
            (abs(charges["amount"] - abs_amount) < 0.01)
        ]
        if match.empty:
            orphans.append(refund)

    if orphans:
        orphan_df = pd.DataFrame(orphans)
        orphan_df["gap_type"] = "Orphan Refund"
        return orphan_df
    return pd.DataFrame()


def detect_rounding_diffs(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Detect sub-cent discrepancies between platform and bank amounts.
    A rounding diff is defined as |diff| > 0 and |diff| < 0.01.
    """
    both = merged[merged["_merge"] == "both"].copy()
    if both.empty:
        return pd.DataFrame()

    both["amount_diff"] = (both["amount_bank"] - both["amount_platform"]).round(4)
    rounding = both[
        (both["amount_diff"].abs() > 0) &
        (both["amount_diff"].abs() < 0.01)
    ].copy()

    if not rounding.empty:
        rounding["gap_type"] = "Rounding Difference"
    return rounding


# ═══════════════════════════════════════════════════════════════════════
#  Classification
# ═══════════════════════════════════════════════════════════════════════

def classify_records(merged: pd.DataFrame,
                     late: pd.DataFrame,
                     rounding: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each record in the merged dataset as:
    - Matched: perfect match, same amounts
    - Gap: missing in one dataset
    - Partial Match: matched but with rounding or late settlement
    """
    classified = merged.copy()
    classified["classification"] = LABEL_MATCHED

    # Gaps = left_only or right_only
    classified.loc[
        classified["_merge"].isin(["left_only", "right_only"]),
        "classification"
    ] = LABEL_GAP

    # Partial = late or rounding
    if not late.empty:
        late_ids = set(late["transaction_id"].tolist())
        classified.loc[
            classified["transaction_id"].isin(late_ids),
            "classification"
        ] = LABEL_PARTIAL

    if not rounding.empty:
        rounding_ids = set(rounding["transaction_id"].tolist())
        classified.loc[
            classified["transaction_id"].isin(rounding_ids),
            "classification"
        ] = LABEL_PARTIAL

    return classified


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

def build_summary(platform_df, bank_df, classified, missing,
                  late, duplicates, orphans, rounding) -> dict:
    """Build a summary dict of all reconciliation metrics."""
    platform_total = platform_df["amount"].sum()
    bank_total = bank_df["amount"].sum()
    net_diff = round(bank_total - platform_total, 4)

    matched_count = (classified["classification"] == LABEL_MATCHED).sum()
    partial_count = (classified["classification"] == LABEL_PARTIAL).sum()
    gap_count = (classified["classification"] == LABEL_GAP).sum()

    return {
        "total_platform_transactions": len(platform_df),
        "total_bank_settlements": len(bank_df),
        "matched_count": int(matched_count),
        "partial_match_count": int(partial_count),
        "gap_count": int(gap_count),
        "missing_in_bank": len(missing),
        "late_settlements": len(late),
        "duplicate_transactions": len(duplicates),
        "orphan_refunds": len(orphans) if not orphans.empty else 0,
        "rounding_differences": len(rounding),
        "platform_total_amount": round(float(platform_total), 2),
        "bank_total_amount": round(float(bank_total), 2),
        "net_difference": float(net_diff),
        "match_rate": round(
            matched_count / max(len(classified), 1) * 100, 1
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Explanation Engine
# ═══════════════════════════════════════════════════════════════════════

def generate_explanations(results: dict) -> list:
    """
    Generate human-readable explanations for each detected gap.
    Returns a list of explanation dicts.
    """
    explanations = []

    # ── Missing Settlements ─────────────────────────────────────────
    missing = results["missing_in_bank"]
    if not missing.empty:
        for _, row in missing.iterrows():
            explanations.append({
                "type": "Missing Settlement",
                "severity": "High",
                "transaction_id": row["transaction_id"],
                "amount": row.get("amount_platform", row.get("amount", 0)),
                "explanation": (
                    f"Transaction {row['transaction_id']} was recorded on the "
                    f"platform on {_fmt_date(row.get('transaction_date', 'N/A'))} "
                    f"for ${abs(row.get('amount_platform', row.get('amount', 0))):.2f}, "
                    f"but no corresponding bank settlement was found. "
                    f"This may indicate a failed payment, a processing delay "
                    f"beyond the normal 1–2 day window, or a bank-side error."
                ),
            })

    # ── Late Settlements ────────────────────────────────────────────
    late = results["late_settlements"]
    if not late.empty:
        for _, row in late.iterrows():
            explanations.append({
                "type": "Late Settlement",
                "severity": "Medium",
                "transaction_id": row["transaction_id"],
                "amount": row.get("amount_platform", 0),
                "explanation": (
                    f"Transaction {row['transaction_id']} was recorded on "
                    f"{_fmt_date(row['transaction_date'])} but settled on "
                    f"{_fmt_date(row['settlement_date'])} "
                    f"({int(row['delay_days'])} days later). "
                    f"This caused a temporary mismatch in the January "
                    f"reconciliation because the settlement crossed into "
                    f"the next month."
                ),
            })

    # ── Duplicates ──────────────────────────────────────────────────
    duplicates = results["duplicates"]
    if not duplicates.empty:
        seen = set()
        for _, row in duplicates.iterrows():
            tid = row["transaction_id"]
            if tid in seen:
                continue
            seen.add(tid)
            count = len(duplicates[duplicates["transaction_id"] == tid])
            explanations.append({
                "type": "Duplicate",
                "severity": "High",
                "transaction_id": tid,
                "amount": row["amount"],
                "explanation": (
                    f"Transaction {tid} appears {count} times in the platform "
                    f"records. The duplicate entry of ${abs(row['amount']):.2f} "
                    f"inflates the platform total and must be investigated — "
                    f"it could be a system glitch or a genuine double-charge "
                    f"that requires a refund to the customer."
                ),
            })

    # ── Orphan Refunds ──────────────────────────────────────────────
    orphans = results["orphan_refunds"]
    if not orphans.empty:
        for _, row in orphans.iterrows():
            explanations.append({
                "type": "Orphan Refund",
                "severity": "High",
                "transaction_id": row["transaction_id"],
                "amount": row["amount"],
                "explanation": (
                    f"Refund {row['transaction_id']} for "
                    f"${abs(row['amount']):.2f} on "
                    f"{_fmt_date(row['transaction_date'])} has no matching "
                    f"original charge from the same customer "
                    f"({row['customer_name']}). This could indicate a "
                    f"fraudulent refund, a data-entry error, or a refund "
                    f"for a transaction from a prior reporting period."
                ),
            })

    # ── Rounding Differences ────────────────────────────────────────
    rounding = results["rounding_diffs"]
    if not rounding.empty:
        for _, row in rounding.iterrows():
            explanations.append({
                "type": "Rounding Difference",
                "severity": "Low",
                "transaction_id": row["transaction_id"],
                "amount": row["amount_diff"],
                "explanation": (
                    f"Transaction {row['transaction_id']} shows a sub-cent "
                    f"discrepancy: platform recorded "
                    f"${row['amount_platform']:.2f} while the bank settled "
                    f"${row['amount_bank']:.4f} (difference: "
                    f"${row['amount_diff']:.4f}). This arises from "
                    f"floating-point arithmetic during batch processing "
                    f"and is typically immaterial."
                ),
            })

    return explanations


def _fmt_date(dt) -> str:
    """Format a date/datetime for display."""
    if pd.isna(dt):
        return "N/A"
    if isinstance(dt, str):
        return dt
    try:
        return pd.Timestamp(dt).strftime("%b %d, %Y")
    except Exception:
        return str(dt)


# ═══════════════════════════════════════════════════════════════════════
#  Console Report
# ═══════════════════════════════════════════════════════════════════════

def print_report(results: dict) -> None:
    """Print a formatted console summary report."""
    s = results["summary"]

    print("\n" + "=" * 68)
    print("  RECONCILIATION SUMMARY REPORT — January 2026")
    print("=" * 68)
    print(f"  Platform Transactions  : {s['total_platform_transactions']}")
    print(f"  Bank Settlements       : {s['total_bank_settlements']}")
    print(f"  Matched                : {s['matched_count']}")
    print(f"  Partial Matches        : {s['partial_match_count']}")
    print(f"  Gaps                   : {s['gap_count']}")
    print("-" * 68)
    print(f"  Missing Settlements    : {s['missing_in_bank']}")
    print(f"  Late Settlements       : {s['late_settlements']}")
    print(f"  Duplicate Transactions : {s['duplicate_transactions']}")
    print(f"  Orphan Refunds         : {s['orphan_refunds']}")
    print(f"  Rounding Differences   : {s['rounding_differences']}")
    print("-" * 68)
    print(f"  Platform Total         : ${s['platform_total_amount']:,.2f}")
    print(f"  Bank Total             : ${s['bank_total_amount']:,.2f}")
    print(f"  Net Difference         : ${s['net_difference']:,.4f}")
    print(f"  Match Rate             : {s['match_rate']}%")
    print("=" * 68)

    print("\n  DETAILED GAP EXPLANATIONS:")
    print("-" * 68)
    for i, exp in enumerate(results["explanations"], 1):
        print(f"\n  [{i}] {exp['type']} — {exp['transaction_id']}")
        print(f"      Severity: {exp['severity']}")
        print(f"      {exp['explanation']}")
    print("\n" + "=" * 68)


if __name__ == "__main__":
    from data_generation import generate_datasets

    datasets = generate_datasets()
    results = reconcile(
        datasets["platform_df"],
        datasets["bank_df"],
        datasets["metadata"]["reconciliation_period"],
    )
    print_report(results)
