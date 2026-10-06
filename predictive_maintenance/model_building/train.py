import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("predictive-maintenance")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

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

with mlflow.start_run(run_name="best-model-search"):
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True, run_name=f"trial_{i+1}"):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_recall", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_cv_recall", float(results["std_test_score"][i]))

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_recall", float(grid_search.best_score_))
    mlflow.log_metric("scale_pos_weight", float(scale_pos_weight))

    train_pred = best_model.predict(Xtrain)
    test_pred = best_model.predict(Xtest)

    train_report = classification_report(ytrain, train_pred, output_dict=True, zero_division=0)
    test_report = classification_report(ytest, test_pred, output_dict=True, zero_division=0)

    mlflow.log_metrics({
        "train_accuracy": float(train_report["accuracy"]),
        "train_precision": float(train_report["1"]["precision"]),
        "train_recall": float(train_report["1"]["recall"]),
        "train_f1": float(train_report["1"]["f1-score"]),
        "test_accuracy": float(test_report["accuracy"]),
        "test_precision": float(test_report["1"]["precision"]),
        "test_recall": float(test_report["1"]["recall"]),
        "test_f1": float(test_report["1"]["f1-score"]),
    })

    print("Best parameters:", grid_search.best_params_)
    print("\nTest classification report:")
    print(classification_report(ytest, test_pred, zero_division=0))

    model_path = "predictive_maintenance/deployment/model.joblib"
    os.makedirs("predictive_maintenance/deployment", exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to: {model_path}")
