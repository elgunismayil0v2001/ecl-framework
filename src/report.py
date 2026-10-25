"""Fill the results tables in README.md and docs/model_documentation.md from reports/tables/*.csv.

Blocks are marked with <!-- NAME:START --> and <!-- NAME:END -->. Missing tables give a placeholder,
so the command is safe to run at any time and gives the same output when run twice.

    python -m src.report
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PLACEHOLDER = "_Not generated yet. Run the pipeline (see README) and then `make report`._"


def markdown_table(df, fmt="{:.4f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read(tables, name):
    path = Path(tables) / name
    return pd.read_csv(path) if path.exists() else None


def _pct(x):
    return f"{x:.2%}"


def render_results_table(tables):
    pdv = _read(tables, "pd_validation.csv")
    cov = _read(tables, "ecl_coverage_by_stage.csv")
    cap = _read(tables, "capital_by_scenario.csv")
    stress = _read(tables, "stress_test.csv")

    def gini(model):
        if pdv is None:
            return "TBD"
        r = pdv[(pdv["model"] == model) & (pdv["metric"] == "gini")]
        return f"{r['value'].iloc[0]:.3f}" if len(r) else "TBD"

    shares, cover_base, cover_down = "TBD", "TBD", "TBD"
    if cov is not None:
        base = cov[cov["scenario"] == "base"]
        total = base["ead"].sum()
        shares = " / ".join(
            _pct(base.loc[base["stage"] == s, "ead"].sum() / total) for s in (1, 2, 3)
        )
        for name in ("base", "downside"):
            d = cov[cov["scenario"] == name]
            value = _pct(d["ecl"].sum() / d["ead"].sum())
            cover_base, cover_down = (value, cover_down) if name == "base" else (cover_base, value)
    density = "TBD"
    if cap is not None and (cap["scenario"] == "base").any():
        density = _pct(cap.loc[cap["scenario"] == "base", "rwa_density"].iloc[0])
    ratio = "TBD"
    if stress is not None and len(stress):
        s = stress.iloc[0]
        ratio = (
            f"{s['capital_ratio_start']:.1%} to {s['capital_ratio_after']:.1%} ({s['change_pp']:+.2f} pp)"
        )
    rows = [
        ("PD scorecard Gini (out-of-time)", gini("scorecard")),
        ("PD LightGBM Gini (out-of-time)", gini("benchmark")),
        ("Stage 1 / 2 / 3 share of exposure", shares),
        ("ECL / exposure (base scenario)", cover_base),
        ("ECL / exposure (downside scenario)", cover_down),
        ("RWA density (RWA / EAD, base)", density),
        ("Capital ratio, base to downside (assumed 12% start)", ratio),
    ]
    return markdown_table(pd.DataFrame(rows, columns=["Item", "Value"]))


def render_pd_validation(tables):
    pdv = _read(tables, "pd_validation.csv")
    if pdv is None:
        return PLACEHOLDER
    t = pdv.pivot_table(index="metric", columns="model", values="value", aggfunc="first").reset_index()
    rule = pdv.drop_duplicates("metric").set_index("metric")["rule"]
    t["gate"] = t["metric"].map(rule)
    return markdown_table(t)


def render_lifetime(tables):
    c = _read(tables, "lifetime_pd_curves.csv")
    if c is None:
        return PLACEHOLDER
    months = [m for m in (12, 60, 120) if m in set(c["month"])]
    t = c[c["month"].isin(months)].pivot_table(index="scenario", columns="month", values="cum_pd")
    t.columns = [f"{m} months" for m in t.columns]
    return markdown_table((t * 100).round(2).reset_index(), fmt="{:.2f}") + "\n\nCumulative PD in %."


def render_ecl_coverage(tables):
    cov = _read(tables, "ecl_coverage_by_stage.csv")
    if cov is None:
        return PLACEHOLDER
    t = cov.pivot_table(index="scenario", columns="stage", values="coverage")
    t.columns = [f"Stage {int(s)}" for s in t.columns]
    return markdown_table((t * 100).reset_index(), fmt="{:.3f}") + "\n\nCoverage = ECL / EAD in %."


def render_capital(tables):
    cap = _read(tables, "capital_by_scenario.csv")
    sens = _read(tables, "capital_sensitivity.csv")
    if cap is None or sens is None:
        return PLACEHOLDER
    keep = ["scenario", "ead_total", "rwa", "rwa_density", "el_irb", "ecl_total"]
    s = sens[["capital_ratio_start", "capital_ratio_after", "change_pp"]]
    return (
        markdown_table(cap[keep], fmt="{:,.4f}")
        + "\n\nStress test by assumed starting capital ratio:\n\n"
        + markdown_table(s, fmt="{:.4f}")
    )


def render_iv_table(tables):
    t = _read(tables, "pd_iv_table.csv")
    if t is None:
        return PLACEHOLDER
    t = t.assign(note=t["note"].fillna(""), selected=t["selected"].map({True: "yes", False: "no"}))
    return markdown_table(t, fmt="{:.3f}")


def render_lgd_bands(tables):
    t = _read(tables, "lgd_by_ltv_band.csv")
    if t is None:
        return PLACEHOLDER
    t = t.assign(n=t["n"].astype(int))
    note = "Bands of LTV at origination (%). LGD as a fraction of the exposure."
    return markdown_table(t, fmt="{:.3f}") + "\n\n" + note


def render_stage_shares(tables):
    cov = _read(tables, "ecl_coverage_by_stage.csv")
    if cov is None:
        return PLACEHOLDER
    base = cov[cov["scenario"] == "base"].sort_values("stage")
    total = base["ead"].sum()
    t = pd.DataFrame(
        {
            "Stage": base["stage"].astype(int).to_numpy(),
            "EAD": [f"{v:,.0f}" for v in base["ead"]],
            "Share of EAD": [_pct(v / total) for v in base["ead"]],
        }
    )
    return markdown_table(t)


def render_sicr_sensitivity(tables):
    t = _read(tables, "sicr_sensitivity.csv")
    if t is None:
        return PLACEHOLDER
    p = t.pivot_table(index="rel_threshold", columns="abs_threshold", values="stage2_share_ead") * 100
    p.columns = [f"+{c * 100:.2f} pp" for c in p.columns]
    p.index = [f"{r:.1f}x" for r in p.index]
    out = p.reset_index().rename(columns={"index": "PD ratio at least"})
    note = "Stage 2 share of performing exposure in %, by SICR thresholds."
    return markdown_table(out, fmt="{:.2f}") + "\n\n" + note


def render_monitoring(tables):
    bt = _read(tables, "monitoring_backtest_by_quarter.csv")
    fpsi = _read(tables, "monitoring_feature_psi.csv")
    mig = _read(tables, "monitoring_stage_migration.csv")
    if bt is None:
        return PLACEHOLDER
    counts = bt["flag"].value_counts()
    parts = [
        "PD backtest by quarter (quarter-end snapshots): "
        + ", ".join(f"{int(counts.get(f, 0))} {f}" for f in ("green", "amber", "red"))
        + f" out of {len(bt)} quarters."
    ]
    if fpsi is not None:
        parts.append("Feature PSI:\n\n" + markdown_table(fpsi, fmt="{:.3f}"))
    if mig is not None:
        mig = mig.rename(columns={mig.columns[0]: "from stage"})
        table = markdown_table(mig, fmt="{:.4f}")
        parts.append("Stage migration between consecutive months (probabilities):\n\n" + table)
    return "\n\n".join(parts)


def _period(value):
    """A readable month for a period label that is either a timestamp or a plain number."""
    t = pd.to_datetime(value, errors="coerce")
    return str(value) if pd.isna(t) else t.strftime("%Y-%m")


def render_data_source(tables):
    src = _read(tables, "data_source.csv")
    if src is None:
        return PLACEHOLDER
    r = src.iloc[0]
    text = (
        f"Data source: **{r['source']}**. {int(r['loans']):,} loans, {int(r['loan_months']):,} loan-months, "
        f"{_period(r['first_month'])} to {_period(r['last_month'])}."
    )
    if any(w in str(r["source"]).lower() for w in ("simulat", "synthetic")):
        text += "\n\n> These results describe a simulation, not the mortgage market."
    return text


def render_split(tables):
    sp = _read(tables, "pd_split.csv")
    if sp is None:
        return PLACEHOLDER
    r = sp.iloc[0]
    rows = [
        ("Training period", f"{_period(r['train_first_period'])} to {_period(r['train_last_period'])}"),
        ("Gap between training and validation (months)", f"{int(r['gap_periods'])}"),
        ("Out-of-time period", f"{_period(r['oot_first_period'])} to {_period(r['oot_last_period'])}"),
        ("Observations (training / out-of-time)", f"{int(r['n_train']):,} / {int(r['n_oot']):,}"),
        ("12-month default rate (training / out-of-time)",
         f"{r['train_default_rate']:.2%} / {r['oot_default_rate']:.2%}"),
        ("Scorecard features", str(r["features"])),
    ]
    return markdown_table(pd.DataFrame(rows, columns=["Item", "Value"]))


BLOCKS = {
    "DATA_SOURCE": render_data_source,
    "SPLIT": render_split,
    "IV_TABLE": render_iv_table,
    "LGD_BANDS": render_lgd_bands,
    "STAGE_SHARES": render_stage_shares,
    "SICR_SENSITIVITY": render_sicr_sensitivity,
    "MONITORING": render_monitoring,
    "RESULTS": render_results_table,
    "PD_VALIDATION": render_pd_validation,
    "LIFETIME": render_lifetime,
    "ECL_COVERAGE": render_ecl_coverage,
    "CAPITAL": render_capital,
}


def replace_block(text, name, content):
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(\n<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text), True


def update_file(path, tables):
    path = Path(path)
    text = path.read_text()
    updated = []
    for name, render in BLOCKS.items():
        text, found = replace_block(text, name, render(tables))
        if found:
            updated.append(name)
    path.write_text(text)
    return updated


def make_charts(tables, figdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []
    cov = _read(tables, "ecl_coverage_by_stage.csv")
    if cov is not None:
        t = cov.pivot_table(index="stage", columns="scenario", values="coverage") * 100
        ax = t.plot(kind="bar", figsize=(7, 4))
        ax.set_title("ECL coverage by stage and scenario")
        ax.set_ylabel("ECL / EAD (%)")
        ax.set_xlabel("Stage")
        ax.figure.tight_layout()
        paths.append(figdir / "ecl_coverage_by_stage.png")
        ax.figure.savefig(paths[-1], dpi=150)
        plt.close(ax.figure)
    sens = _read(tables, "capital_sensitivity.csv")
    if sens is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sens))
        ax.bar(x - 0.2, sens["capital_ratio_start"] * 100, width=0.4, label="before stress")
        ax.bar(x + 0.2, sens["capital_ratio_after"] * 100, width=0.4, label="after stress")
        ax.set_xticks(x, [f"{r:.0%}" for r in sens["capital_ratio_start"]])
        ax.set_title("Capital ratio before and after the downside scenario")
        ax.set_xlabel("Assumed starting capital ratio")
        ax.set_ylabel("Capital / RWA (%)")
        ax.legend()
        fig.tight_layout()
        paths.append(figdir / "capital_stress.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fill result tables in the README and documentation.")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--files", nargs="*", default=["README.md", "docs/model_documentation.md"])
    args = ap.parse_args(argv)
    for f in args.files:
        if Path(f).exists():
            print(f, "updated blocks:", update_file(f, args.tables))
    for p in make_charts(args.tables, args.figures):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
