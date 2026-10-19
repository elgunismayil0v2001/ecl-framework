"""Validation gates: pass/fail checks any PD model must meet before it is used for ECL.

Thresholds are project assumptions. Justify them in docs/model_documentation.md.
"""
from src.validation import calibration_in_the_large, gini, ks, psi, rank_order_corr

DEFAULT_THRESHOLDS = {
    "gini_min": 0.30,
    "ks_min": 0.20,
    "calibration_rel_max": 0.15,
    "psi_max": 0.25,
    "rank_corr_min": 0.90,
}


def evaluate_pd_model(y_test, p_test, p_train=None, thresholds=None):
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    report = {
        "gini": (gini(y_test, p_test), ">=", th["gini_min"]),
        "ks": (ks(y_test, p_test), ">=", th["ks_min"]),
        "calibration_rel_error": (
            abs(calibration_in_the_large(y_test, p_test)),
            "<=",
            th["calibration_rel_max"],
        ),
        "rank_order_corr": (rank_order_corr(y_test, p_test), ">=", th["rank_corr_min"]),
    }
    if p_train is not None:
        report["psi_train_vs_test"] = (psi(p_train, p_test), "<=", th["psi_max"])
    out = {}
    for name, (value, op, limit) in report.items():
        passed = value >= limit if op == ">=" else value <= limit
        out[name] = {"value": value, "rule": f"{op} {limit}", "passed": bool(passed)}
    out["all_passed"] = all(v["passed"] for k, v in out.items() if k != "all_passed")
    return out


def assert_gates(report):
    failed = {k: v for k, v in report.items() if k != "all_passed" and not v["passed"]}
    if failed:
        lines = [f"{k}: {v['value']:.4f} (needs {v['rule']})" for k, v in failed.items()]
        raise AssertionError("Validation gates failed:\n" + "\n".join(lines))
