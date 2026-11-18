"""Week 10: load a saved COMPAS pipeline and predict every row of a new CSV."""
import argparse
import os
import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Return original id, positive-class probability and binary prediction per input row."""
    id_col = config["diagnostics"]["id_column"]
    if id_col not in df.columns:
        raise ValueError(f"Missing identifier column: {id_col}")
    clean = clean_dataset(df, config["diagnostics"])
    X, _, _ = split_features_target(
        clean, config["data"], config["preprocessing"].get("mnar_indicator_sources", []))
    required = config["preprocessing"]["numeric_features"] + config["preprocessing"]["categorical_features"]
    missing = [col for col in required if col not in X.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")
    probability = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({
        id_col: df[id_col].to_numpy(),
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with a saved fitted pipeline.")
    parser.add_argument("input", help="Input CSV without target")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="Output CSV")
    parser.add_argument("--config", default="config.yaml", help="Matching training config")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = config.get("output", {}).get("model_path", "models/model.joblib")
    pipeline = joblib.load(model_path)  # Only load trusted joblib files.
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")
