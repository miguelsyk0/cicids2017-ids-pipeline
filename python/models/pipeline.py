"""
run_pipeline.py

End-to-end orchestration script: fetches cic_typed from Azure SQL, runs
the full preprocessing chain, and trains/evaluates all three models
(Rule-Based, Isolation Forest, XGBoost) -- each with the preprocessing
branch appropriate to it, per the design decisions made throughout this
project. This is the first script that actually ties data_fetch.py,
preprocessing.py, rule_based_model.py, isolation_forest_model.py, and
xgboost_model.py together into one run.

Run this from the same directory as the other project .py files:
    python run_pipeline.py

Requires a working .env (see .env.example) pointing at the Azure SQL
database -- this script cannot run without real network access to it.

WHY THREE SEPARATE PREPROCESSING BRANCHES, not one shared X_train:
    - Rule-Based:       raw, unscaled features (thresholds only mean
                        something in real units)
    - Isolation Forest: scaled, UNBALANCED, continuous features only
                        (no categoricals/hour_of_day -- confirmed by
                        testing that including them roughly doubles the
                        false positive rate; no SMOTE -- rarity is the
                        signal it depends on)
    - XGBoost:          scaled, one-hot encoded, SMOTE-balanced,
                        integer-encoded labels
Building one shared X_train and slicing it per model is tempting but
wrong -- each model's input needs genuinely different treatment, not
just a subset of the same columns.
"""

import os
import pandas as pd

from data_fetch import fetch_training_data
from preprocessing import (
    engineer_timestamp, split_data, encode_categoricals,
    fit_scaler, transform_scaler, apply_smote, build_smote_strategy,
    encode_labels, save_artifact
)
from rule_based_model import RuleBasedModel
from isolation_forest_model import IsolationForestModel
from xgboost_model import XGBoostModel

CATEGORICAL_COLS = ["protocol", "port_group", "source_day"]
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data(chunksize=200_000):
    print("=" * 60)
    print("STEP 1: Fetching data from cic_typed")
    print("=" * 60)
    chunks = fetch_training_data(chunksize=chunksize)
    if isinstance(chunks, pd.DataFrame):
        df = chunks
    else:
        df = pd.concat(list(chunks), ignore_index=True)
    print(f"Fetched {df.shape[0]:,} rows, {df.shape[1]} columns")
    df = engineer_timestamp(df)
    return df


def run_rule_based(df):
    print("\n" + "=" * 60)
    print("STEP 2: Rule-Based Signature Engine")
    print("=" * 60)

    # Raw, unscaled, named features -- drop binary_label (not this
    # model's target) but keep everything else untouched.
    df_rb = df.drop(columns=["binary_label"])
    X_train, X_test, y_train, y_test = split_data(df_rb, target_col="label")

    encoder = load_artifact_if_exists("label_encoder.joblib")
    if encoder is not None:
        rule_model = RuleBasedModel(class_labels=encoder.classes_.tolist())
        y_test_eval = encoder.transform(y_test)
    else:
        # First run: no encoder saved yet. Use the rule model's own
        # default class list and leave y_test as text -- evaluate()
        # doesn't care as long as y_true/y_pred are in the same space,
        # but predict() always returns encoded ints, so fall back to
        # the model's internal label list for consistency here.
        rule_model = RuleBasedModel()
        label_to_index = {label: idx for idx, label in enumerate(rule_model.class_labels)}
        y_test_eval = y_test.map(label_to_index)

    rule_model.fit(X_train)  # optional threshold auto-tuning
    y_pred = rule_model.predict(X_test)

    metrics = rule_model.evaluate(y_test_eval, y_pred)
    print("Rule-Based metrics:", metrics)

    cm_df = rule_model.get_confusion_matrix(y_test_eval, y_pred)
    report_df = rule_model.get_classification_report(y_test_eval, y_pred)
    rule_model.export_for_powerbi(metrics, cm_df, report_df)

    return metrics


def run_isolation_forest(df):
    print("\n" + "=" * 60)
    print("STEP 3: Isolation Forest")
    print("=" * 60)

    # Target is binary_label, not label -- drop the multiclass label
    # entirely so it can't leak into X.
    df_iso = df.drop(columns=["label"])
    X_train, X_test, y_train, y_test = split_data(df_iso, target_col="binary_label")

    X_train_enc, X_test_enc, dummy_cols = encode_categoricals(X_train, X_test, CATEGORICAL_COLS)

    # Drop dummy columns AND hour_of_day entirely -- not just exclude
    # from scaling. Confirmed by testing this matters for Isolation Forest.
    iso_exclude = dummy_cols + ["hour_of_day"]
    X_train_iso = X_train_enc.drop(columns=iso_exclude)
    X_test_iso = X_test_enc.drop(columns=iso_exclude)

    scaler, X_train_scaled = fit_scaler(X_train_iso)
    X_test_scaled = transform_scaler(scaler, X_test_iso)

    attack_ratio = IsolationForestModel.estimate_contamination(y_train, benign_label="BENIGN")
    print(f"Empirical attack ratio in training data: {attack_ratio:.4f}")

    # contamination='auto' by default -- see isolation_forest_model.py's
    # docstring for the tradeoff vs passing attack_ratio explicitly.
    iso_model = IsolationForestModel(contamination="auto")
    iso_model.train(X_train_scaled)

    y_pred = iso_model.predict_labels(X_test_scaled, benign_label="BENIGN", anomaly_label="ATTACK")
    scores = iso_model.anomaly_score(X_test_scaled)

    metrics = iso_model.evaluate(y_test, y_pred)
    print("Isolation Forest metrics:", metrics)

    cm_df = iso_model.get_confusion_matrix(y_test, y_pred)
    report_df = iso_model.get_classification_report(y_test, y_pred)
    scores_df = pd.DataFrame({
        "true_label": pd.Series(y_test).reset_index(drop=True),
        "predicted_label": y_pred,
        "anomaly_score": scores
    })
    iso_model.export_for_powerbi(metrics, cm_df, report_df, scores_df=scores_df)
    iso_model.save_model(os.path.join(ARTIFACT_DIR, "isolation_forest_model.joblib"))

    return metrics


def run_xgboost(df):
    print("\n" + "=" * 60)
    print("STEP 4: XGBoost")
    print("=" * 60)

    # Target is the multiclass label -- drop binary_label so it can't
    # leak in as a second encoding of the same information.
    df_xgb = df.drop(columns=["binary_label"])
    X_train, X_test, y_train, y_test = split_data(df_xgb, target_col="label")

    X_train_enc, X_test_enc, dummy_cols = encode_categoricals(X_train, X_test, CATEGORICAL_COLS)

    scale_exclude = dummy_cols + ["hour_of_day"]
    scaler, X_train_scaled = fit_scaler(X_train_enc, exclude=scale_exclude)
    X_test_scaled = transform_scaler(
        scaler, X_test_enc,
        columns=[c for c in X_test_enc.columns if c not in scale_exclude]
    )

    # Build the SMOTE strategy from REAL class counts rather than a
    # guessed dict -- inspect it before training so you know what's
    # being oversampled and by how much.
    strategy = build_smote_strategy(y_train, min_samples=2000)
    print("SMOTE strategy (classes being boosted):", strategy)

    X_train_bal, y_train_bal = apply_smote(
        X_train_scaled, y_train,
        variant="borderline",
        # apply_smote's typing expects a string for sampling_strategy.
        # build_smote_strategy() returns a dict or None; to satisfy the
        # signature and static type checkers, pass "auto" unless a
        # string is explicitly provided.
        sampling_strategy=(strategy if isinstance(strategy, str) else "auto"),
        k_neighbors=3
    )

    encoder, y_train_final, y_test_final = encode_labels(y_train_bal, y_test)
    save_artifact(encoder, os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    save_artifact(scaler, os.path.join(ARTIFACT_DIR, "scaler_xgb.joblib"))

    xgb_model = XGBoostModel()
    xgb_model.train(X_train_bal, y_train_final, class_labels=encoder.classes_.tolist())
    y_pred = xgb_model.predict(X_test_scaled)

    xgb_model.save_model(os.path.join(ARTIFACT_DIR, "xgboost_model.joblib"))

    importance_df = xgb_model.get_feature_importance(X_train_bal.columns.tolist())
    print("\nTop 10 important features:")
    print(importance_df.head(10))
    importance_df.to_csv("xgboost_feature_importance.csv", index=False)

    metrics = xgb_model.evaluate(y_test_final, y_pred)
    print("XGBoost metrics:", metrics)

    cm_df = xgb_model.get_confusion_matrix(y_test_final, y_pred)
    report_df = xgb_model.get_classification_report(y_test_final, y_pred)
    xgb_model.export_for_powerbi(metrics, cm_df, report_df)

    return metrics


def load_artifact_if_exists(filename):
    from preprocessing import load_artifact
    path = os.path.join(ARTIFACT_DIR, filename)
    if os.path.exists(path):
        return load_artifact(path)
    return None


def build_comparison_table(rb_metrics, iso_metrics, xgb_metrics):
    """
    Pulls all three models' metrics into one table for the BRD / Power BI
    Model Performance page. Note Isolation Forest's metrics are on a
    different task (binary) than Rule-Based/XGBoost (multiclass) -- keep
    that distinction visible rather than implying a false apples-to-apples
    comparison.
    """
    rows = [
        {**rb_metrics, "Task": "Multiclass (attack type)"},
        {**iso_metrics, "Task": "Binary (anomaly detection)"},
        {**xgb_metrics, "Task": "Multiclass (attack type)"},
    ]
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv("model_comparison_summary.csv", index=False)
    print("\nExported: model_comparison_summary.csv")
    return comparison_df


if __name__ == "__main__":
    df = load_data(chunksize=200_000)

    # XGBoost runs FIRST so its saved label_encoder is available for
    # run_rule_based() to reuse -- keeping label order consistent across
    # models, per the single-source-of-truth fix in rule_based_model.py.
    xgb_metrics = run_xgboost(df)
    rb_metrics = run_rule_based(df)
    iso_metrics = run_isolation_forest(df)

    comparison_df = build_comparison_table(rb_metrics, iso_metrics, xgb_metrics)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(comparison_df)