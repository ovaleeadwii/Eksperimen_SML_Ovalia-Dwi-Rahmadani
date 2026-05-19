import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV


# =========================
# 1. Hubungkan ke DagsHub
# =========================

dagshub.init(
    repo_owner="ovaleeadwii",
    repo_name="Eksperimen_SML_Ovalia-Dwi-Rahmadani",
    mlflow=True
)


# =========================
# 2. Path Dataset
# =========================

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "namadataset_preprocessing")

train_path = os.path.join(data_dir, "train_data.csv")
test_path = os.path.join(data_dir, "test_data.csv")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

target_column = "Outcome"

X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]


# =========================
# 3. MLflow Experiment Online
# =========================

mlflow.set_experiment("Diabetes Classification DagsHub Advanced")


# =========================
# 4. Hyperparameter Tuning
# =========================

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5]
}

base_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)


# =========================
# 5. Training + Manual Logging
# =========================

with mlflow.start_run(run_name="RandomForest_DagsHub_Manual_Tuning"):
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    best_cv_score = grid_search.best_score_

    # Manual logging parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("cv", 3)
    mlflow.log_param("scoring", "accuracy")

    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    # Manual logging metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("best_cv_score", best_cv_score)

    # Log model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    # Artifact tambahan 1: model pkl
    model_pkl_path = os.path.join(base_dir, "random_forest_model_dagshub.pkl")
    joblib.dump(best_model, model_pkl_path)
    mlflow.log_artifact(model_pkl_path)

    # Artifact tambahan 2: confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    confusion_matrix_path = os.path.join(base_dir, "confusion_matrix_dagshub.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    mlflow.log_artifact(confusion_matrix_path)

    # Artifact tambahan 3: classification report
    report = classification_report(y_test, y_pred)
    report_path = os.path.join(base_dir, "classification_report_dagshub.txt")

    with open(report_path, "w") as file:
        file.write(report)

    mlflow.log_artifact(report_path)

    # Artifact tambahan 4: feature importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance_path = os.path.join(base_dir, "feature_importance_dagshub.csv")
    feature_importance.to_csv(feature_importance_path, index=False)
    mlflow.log_artifact(feature_importance_path)

    print("Training DagsHub selesai.")
    print("Best Parameters:", best_params)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Best CV Score:", best_cv_score)
    print("Artifact berhasil dikirim ke DagsHub MLflow.")