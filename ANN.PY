import zipfile
import pandas as pd

zip_path = r"C:\Users\Qamar\Downloads\creditcard.csv.zip"  # your zip file path
extract_folder = r"C:\Users\Qamar\Downloads"  # folder to extract to

# Step 1: Extract the CSV file from the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Step 2: Load the extracted CSV file
csv_path = extract_folder + r"\creditcard.csv"  # extracted csv file path

df = pd.read_csv(csv_path)

print(df.head())  # check data loaded properly
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

# Load dataset

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build ANN model
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=2048, verbose=1)

# Predict
y_train_pred = (model.predict(X_train_scaled) > 0.5).astype("int32")
y_val_pred = (model.predict(X_val_scaled) > 0.5).astype("int32")

# Evaluation
print("TRAIN RESULTS:")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1 Score:", f1_score(y_train, y_train_pred))

print("\nVALIDATION RESULTS:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("F1 Score:", f1_score(y_val, y_val_pred))

# Count actual and correctly predicted 1s
print("\nTRAIN SET: Actual 1s =", sum(y_train), " | Correctly Predicted 1s =", np.sum((y_train.values == 1) & (y_train_pred.flatten() == 1)))
print("VALIDATION SET: Actual 1s =", sum(y_val), " | Correctly Predicted 1s =", np.sum((y_val.values == 1) & (y_val_pred.flatten() == 1)))

# Confusion Matrices
train_cm = confusion_matrix(y_train, y_train_pred)
val_cm = confusion_matrix(y_val, y_val_pred)

def plot_conf_matrix(cm, title):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(train_cm, "Train Confusion Matrix")
plot_conf_matrix(val_cm, "Validation Confusion Matrix")

# Metrics Bar Plot
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

df_metrics = pd.DataFrame([train_metrics, val_metrics], index=["Train", "Validation"])
df_metrics.plot(kind="bar", figsize=(8, 5), colormap="Set2")
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Prediction Distribution on Validation
plt.figure(figsize=(6, 3))
sns.countplot(x=y_val_pred.flatten(), palette="pastel")
plt.title("Predicted Class Distribution (Validation Set)")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.xticks([0, 1])
plt.show()
