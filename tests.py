"""
Test Cases for Payments Reconciliation System.

Validates each gap detection function independently using
controlled, minimal datasets.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from reconciliation import (
    detect_duplicates,
    detect_missing_settlements,
    detect_late_settlements,
    detect_orphan_refunds,
    detect_rounding_diffs,
    classify_records,
    reconcile,
)


def _make_platform(records: list) -> pd.DataFrame:
    """Helper: build a platform DataFrame from dicts."""
    df = pd.DataFrame(records)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def _make_bank(records: list) -> pd.DataFrame:
    """Helper: build a bank DataFrame from dicts."""
    df = pd.DataFrame(records)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Test: Duplicate Detection
# ═══════════════════════════════════════════════════════════════════════

def test_duplicate_detection():
    """Duplicated transaction_ids in platform should be flagged."""
    platform = _make_platform([
        {"transaction_id": "TXN-001", "transaction_date": "2026-01-10",
         "customer_name": "Alice", "amount": 100.0, "type": "charge"},
        {"transaction_id": "TXN-001", "transaction_date": "2026-01-10",
         "customer_name": "Alice", "amount": 100.0, "type": "charge"},
        {"transaction_id": "TXN-002", "transaction_date": "2026-01-11",
         "customer_name": "Bob", "amount": 200.0, "type": "charge"},
    ])
    dups = detect_duplicates(platform)
    assert len(dups) == 2, f"Expected 2 duplicate rows, got {len(dups)}"
    assert dups["transaction_id"].unique().tolist() == ["TXN-001"]
    print("  [PASS] test_duplicate_detection")


# ═══════════════════════════════════════════════════════════════════════
#  Test: Missing Settlement Detection
# ═══════════════════════════════════════════════════════════════════════

def test_missing_settlement():
    """Platform txn with no bank match should be flagged."""
    platform = _make_platform([
        {"transaction_id": "TXN-001", "transaction_date": "2026-01-10",
         "customer_name": "Alice", "amount": 100.0, "type": "charge"},
        {"transaction_id": "TXN-002", "transaction_date": "2026-01-12",
         "customer_name": "Bob", "amount": 500.0, "type": "charge"},
    ])
    bank = _make_bank([
        {"settlement_id": "STL-001", "transaction_id": "TXN-001",
         "settlement_date": "2026-01-11", "amount": 100.0},
    ])
    merged = pd.merge(platform, bank, on="transaction_id", how="outer",
                       suffixes=("_platform", "_bank"), indicator=True)
    missing = detect_missing_settlements(merged)
    assert len(missing) == 1, f"Expected 1 missing, got {len(missing)}"
    assert missing.iloc[0]["transaction_id"] == "TXN-002"
    print("  [PASS] test_missing_settlement")


# ═══════════════════════════════════════════════════════════════════════
#  Test: Late Settlement Detection
# ═══════════════════════════════════════════════════════════════════════

def test_late_settlement():
    """Transaction settled in next month should be flagged as late."""
    platform = _make_platform([
        {"transaction_id": "TXN-001", "transaction_date": "2026-01-31",
         "customer_name": "Alice", "amount": 1500.0, "type": "charge"},
    ])
    bank = _make_bank([
        {"settlement_id": "STL-001", "transaction_id": "TXN-001",
         "settlement_date": "2026-02-02", "amount": 1500.0},
    ])
    merged = pd.merge(platform, bank, on="transaction_id", how="outer",
                       suffixes=("_platform", "_bank"), indicator=True)
    late = detect_late_settlements(merged, "2026-01")
    assert len(late) == 1, f"Expected 1 late, got {len(late)}"
    assert int(late.iloc[0]["delay_days"]) == 2
    print("  [PASS] test_late_settlement")


# ═══════════════════════════════════════════════════════════════════════
#  Test: Orphan Refund Detection
# ═══════════════════════════════════════════════════════════════════════

def test_orphan_refund():
    """Refund without a matching original charge should be flagged."""
    platform = _make_platform([
        {"transaction_id": "TXN-REFUND-001", "transaction_date": "2026-01-15",
         "customer_name": "Carol", "amount": -150.0, "type": "refund"},
        {"transaction_id": "TXN-003", "transaction_date": "2026-01-10",
         "customer_name": "David", "amount": 300.0, "type": "charge"},
    ])
    orphans = detect_orphan_refunds(platform)
    assert len(orphans) == 1, f"Expected 1 orphan, got {len(orphans)}"
    assert orphans.iloc[0]["transaction_id"] == "TXN-REFUND-001"
    print("  [PASS] test_orphan_refund")


def test_valid_refund_not_flagged():
    """Refund with a matching charge should NOT be flagged."""
    platform = _make_platform([
        {"transaction_id": "TXN-001", "transaction_date": "2026-01-05",
         "customer_name": "Alice", "amount": 200.0, "type": "charge"},
        {"transaction_id": "TXN-REFUND-001", "transaction_date": "2026-01-10",
         "customer_name": "Alice", "amount": -200.0, "type": "refund"},
    ])
    orphans = detect_orphan_refunds(platform)
    assert len(orphans) == 0, f"Expected 0 orphans, got {len(orphans)}"
    print("  [PASS] test_valid_refund_not_flagged")


# ═══════════════════════════════════════════════════════════════════════
#  Test: Rounding Difference Detection
# ═══════════════════════════════════════════════════════════════════════

def test_rounding_diff():
    """Sub-cent discrepancy should be flagged as rounding."""
    platform = _make_platform([
        {"transaction_id": "TXN-001", "transaction_date": "2026-01-10",
         "customer_name": "Alice", "amount": 112.00, "type": "charge"},
    ])
    bank = _make_bank([
        {"settlement_id": "STL-001", "transaction_id": "TXN-001",
         "settlement_date": "2026-01-11", "amount": 112.005},
    ])
    merged = pd.merge(platform, bank, on="transaction_id", how="outer",
                       suffixes=("_platform", "_bank"), indicator=True)
    rounding = detect_rounding_diffs(merged)
    assert len(rounding) == 1, f"Expected 1 rounding diff, got {len(rounding)}"
    assert abs(rounding.iloc[0]["amount_diff"]) < 0.01
    print("  [PASS] test_rounding_diff")


def test_no_rounding_on_exact_match():
    """Exact match should NOT flag rounding."""
    platform = _make_platform([
        {"transaction_id": "TXN-001", "transaction_date": "2026-01-10",
         "customer_name": "Alice", "amount": 99.99, "type": "charge"},
    ])
    bank = _make_bank([
        {"settlement_id": "STL-001", "transaction_id": "TXN-001",
         "settlement_date": "2026-01-11", "amount": 99.99},
    ])
    merged = pd.merge(platform, bank, on="transaction_id", how="outer",
                       suffixes=("_platform", "_bank"), indicator=True)
    rounding = detect_rounding_diffs(merged)
    assert len(rounding) == 0, f"Expected 0 rounding diffs, got {len(rounding)}"
    print("  [PASS] test_no_rounding_on_exact_match")


# ═══════════════════════════════════════════════════════════════════════
#  Test: Full Reconciliation Pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_full_reconciliation():
    """End-to-end test with multiple gap types."""
    platform = _make_platform([
        {"transaction_id": "TXN-A", "transaction_date": "2026-01-10",
         "customer_name": "Alice", "amount": 100.0, "type": "charge"},
        {"transaction_id": "TXN-B", "transaction_date": "2026-01-20",
         "customer_name": "Bob", "amount": 200.0, "type": "charge"},
        {"transaction_id": "TXN-C", "transaction_date": "2026-01-30",
         "customer_name": "Carol", "amount": 300.0, "type": "charge"},
        {"transaction_id": "TXN-R", "transaction_date": "2026-01-15",
         "customer_name": "Eve", "amount": -50.0, "type": "refund"},
    ])
    bank = _make_bank([
        {"settlement_id": "STL-A", "transaction_id": "TXN-A",
         "settlement_date": "2026-01-11", "amount": 100.0},
        # TXN-B missing
        {"settlement_id": "STL-C", "transaction_id": "TXN-C",
         "settlement_date": "2026-02-01", "amount": 300.0},  # late
        {"settlement_id": "STL-R", "transaction_id": "TXN-R",
         "settlement_date": "2026-01-16", "amount": -50.0},
    ])
    results = reconcile(platform, bank, "2026-01")
    s = results["summary"]

    assert s["missing_in_bank"] == 1, f"Missing: expected 1, got {s['missing_in_bank']}"
    assert s["late_settlements"] == 1, f"Late: expected 1, got {s['late_settlements']}"
    assert s["orphan_refunds"] == 1, f"Orphan: expected 1, got {s['orphan_refunds']}"
    assert len(results["explanations"]) >= 3, "Expected at least 3 explanations"
    print("  [PASS] test_full_reconciliation")


def test_empty_datasets():
    """Empty datasets should produce zero counts."""
    platform = pd.DataFrame({
        "transaction_id": pd.Series(dtype="str"),
        "transaction_date": pd.Series(dtype="datetime64[ns]"),
        "customer_name": pd.Series(dtype="str"),
        "amount": pd.Series(dtype="float64"),
        "type": pd.Series(dtype="str"),
    })
    bank = pd.DataFrame({
        "settlement_id": pd.Series(dtype="str"),
        "transaction_id": pd.Series(dtype="str"),
        "settlement_date": pd.Series(dtype="datetime64[ns]"),
        "amount": pd.Series(dtype="float64"),
    })
    results = reconcile(platform, bank, "2026-01")
    s = results["summary"]
    assert s["total_platform_transactions"] == 0
    assert s["total_bank_settlements"] == 0
    assert s["matched_count"] == 0
    print("  [PASS] test_empty_datasets")


# ═══════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Execute all test functions and report results."""
    tests = [
        test_duplicate_detection,
        test_missing_settlement,
        test_late_settlement,
        test_orphan_refund,
        test_valid_refund_not_flagged,
        test_rounding_diff,
        test_no_rounding_on_exact_match,
        test_full_reconciliation,
        test_empty_datasets,
    ]

    print("\n" + "=" * 56)
    print("  RUNNING TEST SUITE")
    print("=" * 56)

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {test_fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test_fn.__name__} ERROR: {e}")

    print("-" * 56)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 56)
    return passed, failed


if __name__ == "__main__":
    run_all_tests()
