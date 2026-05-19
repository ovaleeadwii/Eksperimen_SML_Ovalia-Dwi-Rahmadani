import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


mlflow.sklearn.autolog()

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

mlflow.set_experiment("Diabetes Classification Basic")

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
