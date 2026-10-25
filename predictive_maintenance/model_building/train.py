import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "predictive_maintenance/deployment/model.joblib"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("predictive-maintenance")

# 1. Load the train and test splits produced by the data-prep job
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# 2. Define the model and the parameter grid
preprocessor = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

negative = int((ytrain == 0).sum())
positive = int((ytrain == 1).sum())
scale_pos_weight = negative / positive if positive else 1.0

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}


def evaluate(model, X, y, prefix):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    metrics = {
        f"{prefix}_accuracy": report["accuracy"],
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
        f"{prefix}_macro_f1": report["macro avg"]["f1-score"],
        f"{prefix}_true_negatives": tn,
        f"{prefix}_false_positives": fp,
        f"{prefix}_false_negatives": fn,
        f"{prefix}_true_positives": tp,
    }
    for cls in ("0", "1"):
        for name in ("precision", "recall", "f1-score"):
            metrics[f"{prefix}_class{cls}_{name.replace('-', '_')}"] = report[cls][name]
    return {k: float(v) for k, v in metrics.items()}


with mlflow.start_run(run_name="best-model-search"):
    # 3. Tune the model
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    # 4. Log every tuned parameter combination as a nested run
    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True, run_name=f"trial_{i+1}"):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_recall", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_cv_recall", float(results["std_test_score"][i]))

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
    mlflow.log_metric("best_cv_recall", float(grid_search.best_score_))

    # 5. Evaluate the best model on the training and test data
    majority_baseline = max(float(ytest.mean()), 1 - float(ytest.mean()))
    mlflow.log_metric("majority_baseline_accuracy", majority_baseline)
    mlflow.log_metrics(evaluate(best_model, Xtrain, ytrain, "train"))
    mlflow.log_metrics(evaluate(best_model, Xtest, ytest, "test"))

    test_pred = best_model.predict(Xtest)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation recall (class 1):", round(grid_search.best_score_, 4))
    print("Majority-class baseline accuracy:", round(majority_baseline, 4))
    print("\nTest classification report:")
    print(classification_report(ytest, test_pred, zero_division=0))

    # 6. Save the best model so the workflow can commit it to the repository
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    print(f"Model saved to: {MODEL_PATH}")
