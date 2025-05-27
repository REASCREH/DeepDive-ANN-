import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# === Paths ===
zip_path = r"C:\Users\Qamar\Downloads\creditcard.csv.zip"
extract_folder = r"C:\Users\Qamar\Downloads"
results_folder = r"C:\Users\Qamar\Downloads\model_results"
os.makedirs(results_folder, exist_ok=True)
results_file = os.path.join(results_folder, "model_results.txt")

# === Step 1: Extract the CSV file ===
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

csv_path = os.path.join(extract_folder, "creditcard.csv")

# === Step 2: Load CSV ===
df = pd.read_csv(csv_path)

# === Data Preprocessing ===
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# === ANN Model ===
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=10, batch_size=2048, verbose=1)

# === Predictions ===
y_train_pred = (model.predict(X_train_scaled) > 0.5).astype("int32")
y_val_pred = (model.predict(X_val_scaled) > 0.5).astype("int32")

# === Metrics ===
train_metrics = {
    "Accuracy": accuracy_score(y_train, y_train_pred),
    "Precision": precision_score(y_train, y_train_pred),
    "Recall": recall_score(y_train, y_train_pred),
    "F1 Score": f1_score(y_train, y_train_pred)
}
val_metrics = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(y_val, y_val_pred),
    "Recall": recall_score(y_val, y_val_pred),
    "F1 Score": f1_score(y_val, y_val_pred)
}

# === Confusion Matrices ===
train_cm = confusion_matrix(y_train, y_train_pred)
val_cm = confusion_matrix(y_val, y_val_pred)

# === Plotting Functions ===
def plot_conf_matrix(cm, title, filename):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()

def plot_metrics_bar(train_metrics, val_metrics, filename):
    df_metrics = pd.DataFrame([train_metrics, val_metrics], index=["Train", "Validation"])
    df_metrics.plot(kind="bar", figsize=(8, 5), colormap="Set2")
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()

def plot_prediction_distribution(y_pred, filename):
    plt.figure(figsize=(6, 3))
    sns.countplot(x=y_pred.flatten(), palette="pastel")
    plt.title("Predicted Class Distribution (Validation Set)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()

# === Generate & Save All Graphs ===
plot_conf_matrix(train_cm, "Train Confusion Matrix", "train_conf_matrix.png")
plot_conf_matrix(val_cm, "Validation Confusion Matrix", "val_conf_matrix.png")
plot_metrics_bar(train_metrics, val_metrics, "metrics_bar.png")
plot_prediction_distribution(y_val_pred, "val_prediction_distribution.png")

# === Save Results to Text File ===
with open(results_file, "w", encoding="utf-8") as f:
    f.write("====== MODEL EVALUATION RESULTS ======\n\n")

    f.write("MODEL SUMMARY:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n")

    f.write("TRAINING METRICS:\n")
    for k, v in train_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

    f.write("\nVALIDATION METRICS:\n")
    for k, v in val_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

    f.write("\nTRAIN CONFUSION MATRIX:\n")
    f.write(np.array2string(train_cm) + "\n")

    f.write("VALIDATION CONFUSION MATRIX:\n")
    f.write(np.array2string(val_cm) + "\n")

    f.write("\nTRAIN SET: Actual frauds = %d | Correctly predicted frauds = %d\n" %
            (sum(y_train), np.sum((y_train.values == 1) & (y_train_pred.flatten() == 1))))

    f.write("VALIDATION SET: Actual frauds = %d | Correctly predicted frauds = %d\n" %
            (sum(y_val), np.sum((y_val.values == 1) & (y_val_pred.flatten() == 1))))

# === Optional: Save Model (if needed) ===
# model.save(os.path.join(results_folder, "fraud_model.h5"))

print(f"\nAll results saved in: {results_folder}")
