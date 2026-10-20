-- DuckDB. Simplified schema: map the columns below to the official Freddie Mac user guide.
-- Raw files are pipe-delimited without headers, so name the columns explicitly when loading.

CREATE OR REPLACE TABLE loans (
    loan_id        VARCHAR,
    orig_date      DATE,
    orig_upb       DOUBLE,
    orig_ltv       DOUBLE,
    orig_dti       DOUBLE,
    credit_score   INTEGER,
    orig_rate      DOUBLE,
    term_months    INTEGER,
    property_state VARCHAR
);

CREATE OR REPLACE TABLE performance (
    loan_id        VARCHAR,
    report_month   DATE,
    current_upb    DOUBLE,
    dpd_months     INTEGER,   -- months delinquent
    zero_bal_code  VARCHAR,   -- liquidation / prepayment flag
    zero_bal_removal_upb DOUBLE,  -- UPB when the loan left the pool (LGD denominator)
    actual_loss    DOUBLE     -- realised loss; positive = loss, negative = gain (check the user guide)
);

CREATE OR REPLACE TABLE macro (
    month          DATE,
    scenario       VARCHAR,   -- base, upside, downside
    unemployment   DOUBLE,
    hpi_yoy        DOUBLE,
    gdp_yoy        DOUBLE
);

CREATE OR REPLACE TABLE scenario_weights (
    scenario VARCHAR,
    weight   DOUBLE
);
INSERT INTO scenario_weights VALUES ('base', 0.6), ('upside', 0.1), ('downside', 0.3);

-- Example load (adjust path and column list):
-- INSERT INTO loans SELECT * FROM read_csv('data/raw/orig.txt', delim='|', header=false, columns={...});
