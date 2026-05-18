import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(input_path):
    """Membaca dataset mentah."""
    data = pd.read_csv(input_path)
    return data


def clean_data(data):
    """Membersihkan data duplikat."""
    data_clean = data.copy()
    data_clean = data_clean.drop_duplicates()
    return data_clean


def preprocess_data(data, target_column):
    """Melakukan split data dan scaling fitur."""

    # Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Bagi data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling fitur numerik
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ubah hasil scaling menjadi dataframe lagi
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Gabungkan fitur dan target
    train_data = X_train_scaled.copy()
    train_data[target_column] = y_train.reset_index(drop=True)

    test_data = X_test_scaled.copy()
    test_data[target_column] = y_test.reset_index(drop=True)

    # Gabungkan train dan test sebagai dataset preprocessing lengkap
    full_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    return train_data, test_data, full_data


def save_data(train_data, test_data, full_data, output_dir):
    """Menyimpan hasil preprocessing ke folder output."""

    os.makedirs(output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
    full_data.to_csv(os.path.join(output_dir, "diabetes_preprocessing.csv"), index=False)


def run_preprocessing():
    """Menjalankan seluruh proses preprocessing."""

    # Lokasi file automate ini
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Path dataset mentah
    input_path = os.path.join(base_dir, "..", "namadataset_raw", "diabetes.csv")

    # Path folder hasil preprocessing
    output_dir = os.path.join(base_dir, "namadataset_preprocessing")

    # Nama kolom target
    target_column = "Outcome"

    print("Mulai preprocessing...")

    data = load_data(input_path)
    print("Dataset berhasil dibaca.")
    print("Ukuran data awal:", data.shape)

    data_clean = clean_data(data)
    print("Data berhasil dibersihkan.")
    print("Ukuran data setelah cleaning:", data_clean.shape)

    train_data, test_data, full_data = preprocess_data(data_clean, target_column)
    print("Data berhasil diproses.")

    save_data(train_data, test_data, full_data, output_dir)
    print("Dataset hasil preprocessing berhasil disimpan.")

    print("Train data:", train_data.shape)
    print("Test data:", test_data.shape)
    print("Full preprocessed data:", full_data.shape)


if __name__ == "__main__":
    run_preprocessing()