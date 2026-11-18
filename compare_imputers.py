"""Week 10 A1: fair 5-fold CV comparison of four imputers, no tuning."""
from copy import deepcopy
import os
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test
from src.model import build_pipeline
from src.evaluate import build_scorers, cross_validate_pipeline, corrected_paired_test


def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    raw = load_data(config["data"]["path"])
    clean = drop_duplicate_rows(clean_dataset(raw, config["diagnostics"]),
                                config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(clean, config["data"],
                                          config["preprocessing"].get("mnar_indicator_sources", []))
    X_dev, _, y_dev, _, _, _ = split_dev_test(
        X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"])
    cv_cfg = config["cv"]
    cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg.get("shuffle", True),
                         random_state=cv_cfg.get("random_state") if cv_cfg.get("shuffle", True) else None)
    scorers = build_scorers(cv_cfg["metrics"])
    metric = cv_cfg["selection_metric"]
    methods = ["median", "knn", "iterative", "random_forest"]
    results = {}
    for method in methods:
        cfg = deepcopy(config["preprocessing"])
        cfg["imputation"]["numeric_strategy"] = method
        print(f"Evaluating {method}...", flush=True)
        pipeline = build_pipeline(cfg, config["model"], config.get("feature_selection"))
        scores, _ = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, metric,
                                             n_jobs=cv_cfg.get("n_jobs", 1))
        results[method] = scores
        print(f"  {metric}: {scores[metric].mean():.3f} ± {scores[metric].std(ddof=1):.3f}")
    summary = pd.DataFrame([{"imputer": name, **{m: scores[m].mean() for m in scorers}}
                            for name, scores in results.items()])
    ratio = 1 / (cv_cfg["n_splits"] - 1)
    tests = pd.DataFrame([{"comparison": f"{name} vs median", **corrected_paired_test(
        results[name][metric], results["median"][metric], ratio)}
        for name in methods if name != "median"])
    os.makedirs("results/week10", exist_ok=True)
    summary.to_csv("results/week10/imputer_scores.csv", index=False)
    tests.to_csv("results/week10/imputer_corrected_tests.csv", index=False)
    print("\nMean CV metrics:\n", summary.to_string(index=False))
    print("\nCorrected paired tests (difference = method - median):\n", tests.to_string(index=False))
    print("\nSaved results to results/week10/")


if __name__ == "__main__":
    main()
