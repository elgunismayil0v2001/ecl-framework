"""Load Freddie Mac origination and performance files into the `loans` and `performance` tables.

The file layout (v1 or v2, see src/freddie_schema.py) is detected from the number of fields.
`--orig` and `--perf` accept one or more glob patterns separated by commas.

Cleaning rules (assumptions, document them):
- orig_date = first payment month minus one month (the file has no origination date)
- credit score 9999, LTV 999 and DTI 999 mean "not available" and become NULL
- interest rate is converted from percent to a fraction
- delinquency status: number of months past due (zero padded in v2); 'RA' (REO acquisition) is set to 3
  (default); 'XX' (unknown) rows are dropped
- realised loss is converted to "positive = loss" (v1 files are multiplied by -1)
- credit-event zero balance codes (short sale, REO, ...) are kept in zero_bal_code but do not
  change the 90+ DPD default definition
"""
import glob
import os

from src.freddie_schema import LAYOUTS, count_fields, detect_layout, internal_names


def _columns_struct(columns):
    return "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in columns) + "}"


def _quote(text):
    return text.replace("'", "''")


def expand(spec):
    """Sorted files matching one or more comma-separated glob patterns."""
    patterns = [p.strip() for p in spec.split(",") if p.strip()]
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        folder = os.path.dirname(patterns[0]) if patterns else "."
        folder = folder or "."
        found = sorted(os.listdir(folder))[:10] if os.path.isdir(folder) else "folder does not exist"
        raise FileNotFoundError(
            f"No files match {spec}. Files in {folder}: {found}. "
            "Unzip the Freddie Mac files into that folder, or pass --orig / --perf (ORIG / PERF in make)."
        )
    return files


def check_files(pattern, expected):
    """Files matching `pattern`, each with `expected` fields. Raises a clear error otherwise."""
    files = expand(pattern)
    for f in files:
        n = count_fields(f)
        if n != expected:
            raise ValueError(
                f"{f} has {n} fields, expected {expected}. "
                "Check src/freddie_schema.py against the user guide of your file version."
            )
    return files


def validate_inputs(orig_spec, perf_spec):
    """Find the files, detect the layout and check every file. Returns (orig_files, perf_files, layout)."""
    orig_files, perf_files = expand(orig_spec), expand(perf_spec)
    layout = detect_layout(count_fields(orig_files[0]), count_fields(perf_files[0]))
    for files, key in ((orig_files, "orig"), (perf_files, "perf")):
        expected = len(LAYOUTS[layout][key])
        for f in files:
            n = count_fields(f)
            if n != expected:
                raise ValueError(
                    f"{f} has {n} fields but the {layout} layout needs {expected}. "
                    "Do not mix files from different layouts in one load."
                )
    return orig_files, perf_files, layout


def _file_list(files):
    return "[" + ", ".join("'" + _quote(f) + "'" for f in files) + "]"


def load_freddie(con, orig_spec, perf_spec):
    orig_files, perf_files, layout = validate_inputs(orig_spec, perf_spec)
    orig_columns = internal_names(LAYOUTS[layout]["orig"])
    perf_columns = internal_names(LAYOUTS[layout]["perf"])
    loss_sign = LAYOUTS[layout]["loss_sign"]

    con.execute(
        f"""CREATE OR REPLACE TABLE raw_orig AS
            SELECT * FROM read_csv({_file_list(orig_files)}, delim='|', header=false,
                                   columns={_columns_struct(orig_columns)})"""
    )
    con.execute(
        f"""CREATE OR REPLACE TABLE raw_perf AS
            SELECT * FROM read_csv({_file_list(perf_files)}, delim='|', header=false,
                                   columns={_columns_struct(perf_columns)})"""
    )

    con.execute("DELETE FROM loans")
    con.execute("DELETE FROM performance")

    con.execute(
        """INSERT INTO loans
           SELECT
               loan_sequence_number AS loan_id,
               CAST(make_date(CAST(substr(first_payment_date, 1, 4) AS INTEGER),
                              CAST(substr(first_payment_date, 5, 2) AS INTEGER), 1)
                    - INTERVAL 1 MONTH AS DATE) AS orig_date,
               TRY_CAST(original_upb AS DOUBLE) AS orig_upb,
               NULLIF(TRY_CAST(original_ltv AS DOUBLE), 999) AS orig_ltv,
               NULLIF(TRY_CAST(original_dti AS DOUBLE), 999) AS orig_dti,
               NULLIF(TRY_CAST(credit_score AS INTEGER), 9999) AS credit_score,
               TRY_CAST(original_interest_rate AS DOUBLE) / 100.0 AS orig_rate,
               TRY_CAST(original_loan_term AS INTEGER) AS term_months,
               property_state
           FROM raw_orig
           WHERE loan_sequence_number IS NOT NULL"""
    )

    con.execute(
        f"""INSERT INTO performance
           SELECT loan_id, report_month, current_upb, dpd_months, zero_bal_code,
                  zero_bal_removal_upb, actual_loss
           FROM (
               SELECT
                   loan_sequence_number AS loan_id,
                   make_date(CAST(substr(monthly_reporting_period, 1, 4) AS INTEGER),
                             CAST(substr(monthly_reporting_period, 5, 2) AS INTEGER), 1) AS report_month,
                   TRY_CAST(current_actual_upb AS DOUBLE) AS current_upb,
                   CASE
                       WHEN current_loan_delinquency_status = 'RA' THEN 3
                       ELSE TRY_CAST(current_loan_delinquency_status AS INTEGER)
                   END AS dpd_months,
                   zero_balance_code AS zero_bal_code,
                   TRY_CAST(zero_balance_removal_upb AS DOUBLE) AS zero_bal_removal_upb,
                   TRY_CAST(actual_loss_calculation AS DOUBLE) * {loss_sign} AS actual_loss
               FROM raw_perf
               WHERE loan_sequence_number IS NOT NULL
                 AND monthly_reporting_period IS NOT NULL
           )
           WHERE dpd_months IS NOT NULL"""
    )

    n_loans = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    n_perf = con.execute("SELECT COUNT(*) FROM performance").fetchone()[0]
    return {"loans": n_loans, "performance_rows": n_perf, "layout": layout}
