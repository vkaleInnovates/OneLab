"""
Synthetic Data Generator for Payments Reconciliation System.

Generates realistic platform transactions and bank settlement datasets
with intentional gap scenarios for reconciliation testing.

ASSUMPTIONS:
1. All amounts are in USD (single currency)
2. Platform records transactions instantly on payment date
3. Bank settles transactions 1–2 business days after platform date
4. Transaction IDs follow format TXN-YYYYMMDD-XXXX
5. Settlement IDs follow format STL-YYYYMMDD-XXXX
6. Reconciliation period: January 2026 (month-end)
7. Refunds are negative amounts with type='refund'
8. No partial or split settlements
9. Customer names are randomly sampled from a pool
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string


CUSTOMER_NAMES = [
    "Alice Johnson", "Bob Martinez", "Carol Williams", "David Chen",
    "Elena Rodriguez", "Frank Thompson", "Grace Kim", "Henry Wilson",
    "Irene Patel", "James O'Brien", "Karen Nakamura", "Liam Foster",
    "Maria Santos", "Nathan Hughes", "Olivia Bennett", "Paul Schneider",
    "Quinn Murphy", "Rachel Adams", "Samuel Lee", "Tanya Gupta",
    "Uma Krishnan", "Victor Morales", "Wendy Chang", "Xavier Reyes",
    "Yuki Tanaka", "Zachary Brown",
]

RECON_YEAR = 2026
RECON_MONTH = 1  # January


def _generate_id(prefix: str, date: datetime, seq: int) -> str:
    return f"{prefix}-{date.strftime('%Y%m%d')}-{seq:04d}"


def _random_amount(low: float = 15.0, high: float = 5000.0) -> float:
    return round(random.uniform(low, high), 2)


def generate_datasets(seed: int = 42) -> dict:
    """
    Generate platform transactions and bank settlements with injected gaps.

    Returns:
        dict with keys:
            - platform_df: pd.DataFrame
            - bank_df: pd.DataFrame
            - metadata: dict of injected gap info
    """
    random.seed(seed)
    np.random.seed(seed)

    platform_records = []
    bank_records = []
    txn_seq = 1
    stl_seq = 1

    # Track injected gaps for validation
    injected = {
        "late_settlements": [],
        "rounding_diffs": [],
        "duplicate_txns": [],
        "missing_in_bank": [],
        "orphan_refunds": [],
    }

    # ── 1. Normal matched transactions ──────────────────────────────────
    days_in_month = 31  # January
    for day in range(1, days_in_month + 1):
        txn_date = datetime(RECON_YEAR, RECON_MONTH, day)
        n_txns = random.randint(3, 6)

        for _ in range(n_txns):
            txn_id = _generate_id("TXN", txn_date, txn_seq)
            txn_seq += 1
            amount = _random_amount()
            customer = random.choice(CUSTOMER_NAMES)
            settle_delay = random.randint(1, 2)
            settle_date = txn_date + timedelta(days=settle_delay)

            platform_records.append({
                "transaction_id": txn_id,
                "transaction_date": txn_date.date(),
                "customer_name": customer,
                "amount": amount,
                "type": "charge",
            })

            bank_records.append({
                "settlement_id": _generate_id("STL", settle_date, stl_seq),
                "transaction_id": txn_id,
                "settlement_date": settle_date.date(),
                "amount": amount,
            })
            stl_seq += 1

    # ── 2. Gap: Late settlements (settled in February) ──────────────────
    for i in range(4):
        late_day = random.randint(29, 31)
        txn_date = datetime(RECON_YEAR, RECON_MONTH, late_day)
        txn_id = _generate_id("TXN", txn_date, txn_seq)
        txn_seq += 1
        amount = _random_amount(500, 3000)
        customer = random.choice(CUSTOMER_NAMES)

        # Settle in February (next month)
        feb_day = random.randint(1, 3)
        settle_date = datetime(RECON_YEAR, 2, feb_day)

        platform_records.append({
            "transaction_id": txn_id,
            "transaction_date": txn_date.date(),
            "customer_name": customer,
            "amount": amount,
            "type": "charge",
        })
        bank_records.append({
            "settlement_id": _generate_id("STL", settle_date, stl_seq),
            "transaction_id": txn_id,
            "settlement_date": settle_date.date(),
            "amount": amount,
        })
        stl_seq += 1
        injected["late_settlements"].append(txn_id)

    # ── 3. Gap: Rounding discrepancies ──────────────────────────────────
    # Bank processes with sub-cent precision; platform rounds to 2 decimals
    rounding_indices = random.sample(range(len(bank_records) - 10), 5)
    for idx in rounding_indices:
        original = bank_records[idx]["amount"]
        # Add tiny offset ±0.001 to ±0.009
        offset = round(random.uniform(0.001, 0.009), 4)
        sign = random.choice([-1, 1])
        bank_records[idx]["amount"] = round(original + sign * offset, 4)
        injected["rounding_diffs"].append({
            "txn_id": bank_records[idx]["transaction_id"],
            "platform_amount": original,
            "bank_amount": bank_records[idx]["amount"],
            "diff": round(sign * offset, 4),
        })

    # ── 4. Gap: Duplicate transactions in platform ──────────────────────
    for i in range(3):
        src_idx = random.randint(0, min(60, len(platform_records) - 1))
        duplicate = platform_records[src_idx].copy()
        platform_records.append(duplicate)
        injected["duplicate_txns"].append(duplicate["transaction_id"])

    # ── 5. Gap: Missing in bank (platform has, bank doesn't) ───────────
    for i in range(4):
        day = random.randint(5, 25)
        txn_date = datetime(RECON_YEAR, RECON_MONTH, day)
        txn_id = _generate_id("TXN", txn_date, txn_seq)
        txn_seq += 1
        amount = _random_amount(100, 2000)
        customer = random.choice(CUSTOMER_NAMES)

        platform_records.append({
            "transaction_id": txn_id,
            "transaction_date": txn_date.date(),
            "customer_name": customer,
            "amount": amount,
            "type": "charge",
        })
        # NOT adding to bank_records
        injected["missing_in_bank"].append(txn_id)

    # ── 6. Gap: Orphan refunds (no matching original) ───────────────────
    for i in range(3):
        day = random.randint(10, 28)
        txn_date = datetime(RECON_YEAR, RECON_MONTH, day)
        refund_id = _generate_id("TXN", txn_date, txn_seq)
        txn_seq += 1
        refund_amount = -_random_amount(50, 500)
        customer = random.choice(CUSTOMER_NAMES)

        # Fabricate a non-existent original reference
        fake_original = f"TXN-20251215-{9000 + i:04d}"

        platform_records.append({
            "transaction_id": refund_id,
            "transaction_date": txn_date.date(),
            "customer_name": customer,
            "amount": refund_amount,
            "type": "refund",
        })

        settle_date = txn_date + timedelta(days=1)
        bank_records.append({
            "settlement_id": _generate_id("STL", settle_date, stl_seq),
            "transaction_id": refund_id,
            "settlement_date": settle_date.date(),
            "amount": refund_amount,
        })
        stl_seq += 1

        injected["orphan_refunds"].append({
            "refund_id": refund_id,
            "fake_original": fake_original,
            "amount": refund_amount,
        })

    # ── Build DataFrames ────────────────────────────────────────────────
    platform_df = pd.DataFrame(platform_records)
    platform_df["transaction_date"] = pd.to_datetime(platform_df["transaction_date"])
    platform_df = platform_df.sort_values("transaction_date").reset_index(drop=True)

    bank_df = pd.DataFrame(bank_records)
    bank_df["settlement_date"] = pd.to_datetime(bank_df["settlement_date"])
    bank_df = bank_df.sort_values("settlement_date").reset_index(drop=True)

    return {
        "platform_df": platform_df,
        "bank_df": bank_df,
        "metadata": {
            "reconciliation_period": f"{RECON_YEAR}-{RECON_MONTH:02d}",
            "total_platform": len(platform_df),
            "total_bank": len(bank_df),
            "injected_gaps": injected,
        },
    }


def print_summary(datasets: dict) -> None:
    """Print a console summary of generated datasets."""
    meta = datasets["metadata"]
    gaps = meta["injected_gaps"]

    print("=" * 64)
    print("  SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 64)
    print(f"  Reconciliation Period : {meta['reconciliation_period']}")
    print(f"  Platform Transactions : {meta['total_platform']}")
    print(f"  Bank Settlements      : {meta['total_bank']}")
    print("-" * 64)
    print("  Injected Gap Scenarios:")
    print(f"    Late Settlements    : {len(gaps['late_settlements'])}")
    print(f"    Rounding Diffs      : {len(gaps['rounding_diffs'])}")
    print(f"    Duplicate Txns      : {len(gaps['duplicate_txns'])}")
    print(f"    Missing in Bank     : {len(gaps['missing_in_bank'])}")
    print(f"    Orphan Refunds      : {len(gaps['orphan_refunds'])}")
    print("=" * 64)


if __name__ == "__main__":
    datasets = generate_datasets()
    print_summary(datasets)
    print("\nPlatform sample:")
    print(datasets["platform_df"].head(10).to_string(index=False))
    print("\nBank sample:")
    print(datasets["bank_df"].head(10).to_string(index=False))
