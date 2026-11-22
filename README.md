# Predictive Maintenance - MLOps Pipeline

Binary classification of engine condition (normal vs. maintenance / failure-risk) from six engine sensor readings,
with an automated MLOps workflow: **GitHub Actions + MLflow + Streamlit Community Cloud**.

## Repository structure
```
.github/workflows/pipeline.yml        # CI/CD pipeline (register -> prepare -> train -> commit model)
predictive_maintenance/
  data/engine_data.csv                # raw dataset (19,535 rows, 7 columns)
  model_building/
    data_register.py                  # validates expected columns, prints dataset summary
    prep.py                           # cleaning + stratified 80/20 train/test split
    train.py                          # XGBoost + GridSearchCV + MLflow tracking, saves model.joblib
  deployment/
    app.py                            # Streamlit app
    model.joblib                      # best model, committed automatically by the pipeline
    requirements.txt                  # Streamlit app dependencies
  requirements.txt                    # pipeline (training) dependencies
```

## Pipeline (`.github/workflows/pipeline.yml`)
1. **register-dataset** - checks the expected columns and prints a data summary.
2. **data-prep** - removes duplicates, splits train/test and uploads the splits as the `data-splits` workflow artifact.
3. **model-training** - downloads the artifact, tunes XGBoost (GridSearchCV, recall), logs all parameter combinations and
   metrics to MLflow, saves the best model and commits `model.joblib` back to `main`.

## Results (held-out test set, 3,907 records)
Accuracy 0.645 | Precision (risk class) 0.766 | Recall 0.629 | F1 0.691

## Run locally
```
pip install -r predictive_maintenance/requirements.txt
python predictive_maintenance/model_building/data_register.py
python predictive_maintenance/model_building/prep.py
python predictive_maintenance/model_building/train.py
pip install -r predictive_maintenance/deployment/requirements.txt
streamlit run predictive_maintenance/deployment/app.py
```
