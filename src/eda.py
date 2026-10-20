"""Exploratory analysis: 12-month default rate by vintage, credit score band and LTV band."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import DB_PATH, connect  # noqa: E402

CREDIT_SCORE_EDGES = [620, 660, 700, 740, 780]
LTV_EDGES = [60, 70, 80, 90, 95]


def band_labels(edges):
    labels = [f"<{edges[0]}"]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"{a}-{b - 1}")
    labels.append(f"{edges[-1]}+")
    return labels


def band_case_sql(column, edges):
    whens = " ".join(f"WHEN {column} < {e} THEN {i}" for i, e in enumerate(edges))
    return f"CASE WHEN {column} IS NULL THEN NULL {whens} ELSE {len(edges)} END"


def vintage_default_rate(con):
    return con.execute(
        """SELECT vintage_quarter, COUNT(*) AS observations, AVG(default_12m) AS default_rate_12m
           FROM features GROUP BY vintage_quarter ORDER BY vintage_quarter"""
    ).df()


def default_rate_by_band(con, column, edges):
    labels = band_labels(edges)
    df = con.execute(
        f"""SELECT {band_case_sql(column, edges)} AS band_idx,
                   COUNT(*) AS observations,
                   AVG(default_12m) AS default_rate_12m
            FROM features
            WHERE {column} IS NOT NULL
            GROUP BY 1 ORDER BY 1"""
    ).df()
    df["band"] = df["band_idx"].map(lambda i: labels[int(i)])
    return df


def _bar(df, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["band"], df["default_rate_12m"] * 100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("12-month default rate (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(con, outdir="reports/figures"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    vint = vintage_default_rate(con)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pd.to_datetime(vint["vintage_quarter"]), vint["default_rate_12m"] * 100, marker="o")
    ax.set_title("12-month default rate by origination quarter")
    ax.set_xlabel("Origination quarter")
    ax.set_ylabel("12-month default rate (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    paths.append(outdir / "default_rate_by_vintage.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    score = default_rate_by_band(con, "credit_score", CREDIT_SCORE_EDGES)
    paths.append(outdir / "default_rate_by_credit_score.png")
    _bar(score, "Default rate by credit score band", "Credit score", paths[-1])

    ltv = default_rate_by_band(con, "orig_ltv", LTV_EDGES)
    paths.append(outdir / "default_rate_by_ltv.png")
    _bar(ltv, "Default rate by original LTV band", "Original LTV (%)", paths[-1])

    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Create EDA charts from the DuckDB database.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--outdir", default="reports/figures")
    args = ap.parse_args(argv)
    for p in make_figures(connect(args.db), args.outdir):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
