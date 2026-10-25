"""Column layouts of the Freddie Mac Single-Family Loan-Level Dataset (pipe-delimited, no header).

Two layouts exist:
- v1: files before Release 47 (July 2026): 32 origination and 32 performance fields.
- v2: Standard Dataset files from Release 47: 31 origination and 35 performance fields (orig_YYYYQn.txt and
  perf_YYYYQn.txt). Servicer name and MI cancellation moved to the performance file, VantageScore 4.0 and
  bankruptcy cramdown costs were added, and the sign of the loss fields changed.
The loader detects the layout from the number of fields. Check the lists against the user guide of the
file version you downloaded.
"""

ORIG_COLUMNS_V1 = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa",
    "mi_pct",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming_flag",
    "pre_harp_loan_sequence_number",
    "program_indicator",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
]

PERF_COLUMNS_V1 = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "due_date_of_last_paid_installment",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss_calculation",
    "cumulative_modification_cost",
    "step_modification_flag",
    "payment_deferral",
    "estimated_loan_to_value",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]


def count_fields(path):
    """Number of pipe-separated fields in the first non-empty line of a file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                return len(line.split("|"))
    return 0


# Backward-compatible names for the v1 layout.
ORIG_COLUMNS = ORIG_COLUMNS_V1
PERF_COLUMNS = PERF_COLUMNS_V1

ORIG_COLUMNS_V2 = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_indicator",
    "maturity_date",
    "msa",
    "mi_pct",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_indicator",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_identifier",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_sequence_number",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "vantagescore_4",
]

PERF_COLUMNS_V2 = [
    "loan_identifier",
    "period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "underwriting_defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "due_date_of_last_paid_installment",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "estimated_loan_to_value",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "current_period_modification_costs",
    "current_interest_bearing_upb",
    "mortgage_insurance_cancellation_indicator",
    "servicer_name",
    "bankruptcy_cramdown_costs",
]

# v2 names of the fields the loader uses, mapped to the names used in the loading SQL.
RENAMES = {
    "loan_identifier": "loan_sequence_number",
    "period": "monthly_reporting_period",
    "actual_loss": "actual_loss_calculation",
}

# Sign of the realised loss field. v2: a loss is positive (stated in the Release 47 notes).
# v1: inferred from those notes (the sign "changed"): a loss is negative. Verify with `make quality`
# (the lgd_mean_plausible check) and the user guide of your files.
LAYOUTS = {
    "v1": {"orig": ORIG_COLUMNS_V1, "perf": PERF_COLUMNS_V1, "loss_sign": -1.0},
    "v2": {"orig": ORIG_COLUMNS_V2, "perf": PERF_COLUMNS_V2, "loss_sign": 1.0},
}


def detect_layout(orig_fields, perf_fields):
    """Layout name from the field counts of an origination and a performance file."""
    for name, layout in LAYOUTS.items():
        if (len(layout["orig"]), len(layout["perf"])) == (orig_fields, perf_fields):
            return name
    expected = {n: (len(v["orig"]), len(v["perf"])) for n, v in LAYOUTS.items()}
    raise ValueError(
        f"Unknown file layout: {orig_fields} origination and {perf_fields} performance fields. "
        f"Known layouts (origination, performance): {expected}. If every line ends with a '|', "
        "remove the trailing pipe. Otherwise compare with the user guide and update src/freddie_schema.py."
    )


def internal_names(columns):
    return [RENAMES.get(c, c) for c in columns]
