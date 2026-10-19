"""Load Freddie Mac origination and performance files into the `loans` and `performance` tables.

Cleaning rules (assumptions, document them):
- orig_date = first payment month minus one month (the file has no origination date)
- credit score 9999, LTV 999 and DTI 999 mean "not available" and become NULL
- interest rate is converted from percent to a fraction
- delinquency status: number of months past due; 'RA' (REO acquisition) is set to 3 (default);
  'XX' (unknown) rows are dropped
- credit-event zero balance codes (short sale, REO, ...) are kept in zero_bal_code but do not
  change the 90+ DPD default definition
"""
import glob

from src.freddie_schema import ORIG_COLUMNS, PERF_COLUMNS, count_fields


def _columns_struct(columns):
    return "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in columns) + "}"


def _quote(text):
    return text.replace("'", "''")


def _check_files(pattern, expected):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    for f in files:
        n = count_fields(f)
        if n != expected:
            raise ValueError(
                f"{f} has {n} fields, expected {expected}. "
                "Check src/freddie_schema.py against the user guide of your file version."
            )
    return files


def load_freddie(con, orig_glob, perf_glob):
    _check_files(orig_glob, len(ORIG_COLUMNS))
    _check_files(perf_glob, len(PERF_COLUMNS))

    con.execute(
        f"""CREATE OR REPLACE TABLE raw_orig AS
            SELECT * FROM read_csv('{_quote(orig_glob)}', delim='|', header=false,
                                   columns={_columns_struct(ORIG_COLUMNS)})"""
    )
    con.execute(
        f"""CREATE OR REPLACE TABLE raw_perf AS
            SELECT * FROM read_csv('{_quote(perf_glob)}', delim='|', header=false,
                                   columns={_columns_struct(PERF_COLUMNS)})"""
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
        """INSERT INTO performance
           SELECT loan_id, report_month, current_upb, dpd_months, zero_bal_code
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
                   zero_balance_code AS zero_bal_code
               FROM raw_perf
               WHERE loan_sequence_number IS NOT NULL
                 AND monthly_reporting_period IS NOT NULL
           )
           WHERE dpd_months IS NOT NULL"""
    )

    n_loans = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    n_perf = con.execute("SELECT COUNT(*) FROM performance").fetchone()[0]
    return {"loans": n_loans, "performance_rows": n_perf}
