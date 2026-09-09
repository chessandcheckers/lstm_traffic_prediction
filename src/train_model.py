import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_PATH = "data/traffic.csv"
MODEL_PATH = "models/traffic_lstm.keras"

SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.8


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

df["DateTime"] = pd.to_datetime(df["DateTime"])

df = df.sort_values(
    by=["Junction", "DateTime"]
).reset_index(drop=True)

df = df.dropna(
    subset=["DateTime", "Junction", "Vehicles"]
)


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek
df["Month"] = df["DateTime"].dt.month

print("\nProcessed dataset:")
print(df.head())


# ---------------------------------------------------------
# SELECT ONE JUNCTION
# ---------------------------------------------------------

junction_id = 1

junction_df = df[
    df["Junction"] == junction_id
].copy()

junction_df = junction_df.sort_values(
    "DateTime"
).reset_index(drop=True)

print(f"\nUsing Junction {junction_id}")
print("Number of observations:", len(junction_df))


# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "Vehicles",
    "Hour",
    "DayOfWeek",
    "Month"
]

TARGET_COLUMN = "Vehicles"


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

split_index = int(
    len(junction_df) * TRAIN_RATIO
)

train_df = junction_df.iloc[
    :split_index
].copy()

test_df = junction_df.iloc[
    split_index:
].copy()

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))


# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

feature_scaler = MinMaxScaler()

train_features = feature_scaler.fit_transform(
    train_df[FEATURE_COLUMNS]
)

test_features = feature_scaler.transform(
    test_df[FEATURE_COLUMNS]
)


# ---------------------------------------------------------
# SCALE TARGET
# ---------------------------------------------------------

target_scaler = MinMaxScaler()

train_targets = target_scaler.fit_transform(
    train_df[[TARGET_COLUMN]]
).flatten()

test_targets = target_scaler.transform(
    test_df[[TARGET_COLUMN]]
).flatten()


# ---------------------------------------------------------
# CREATE SEQUENCES
#
# Previous 24 hours -> predict next hour
# ---------------------------------------------------------

def create_sequences(
    features,
    targets,
    sequence_length
):

    X = []
    y = []

    for i in range(
        sequence_length,
        len(features)
    ):

        X.append(
            features[
                i - sequence_length:i
            ]
        )

        y.append(
            targets[i]
        )

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(
    train_features,
    train_targets,
    SEQUENCE_LENGTH
)

X_test, y_test = create_sequences(
    test_features,
    test_targets,
    SEQUENCE_LENGTH
)


print("\nSequence shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ---------------------------------------------------------
# BUILD LSTM MODEL
# ---------------------------------------------------------

model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS)
        )
    ),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)

])


model.compile(
    optimizer="adam",
    loss="mse"
)


print("\nModel architecture:")
model.summary()


# ---------------------------------------------------------
# EARLY STOPPING
# ---------------------------------------------------------

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining model...\n")

history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=50,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ---------------------------------------------------------
# LSTM PREDICTIONS
# ---------------------------------------------------------

predictions_scaled = model.predict(
    X_test
)

predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual_values = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()


# ---------------------------------------------------------
# LSTM EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(
    actual_values,
    predictions
)

mse = mean_squared_error(
    actual_values,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual_values,
    predictions
)


# ---------------------------------------------------------
# NAIVE BASELINE
#
# Predict next-hour traffic using the most recent
# observed vehicle count.
# ---------------------------------------------------------

vehicle_feature_index = FEATURE_COLUMNS.index(
    "Vehicles"
)

baseline_scaled = X_test[
    :,
    -1,
    vehicle_feature_index
]

baseline_predictions = target_scaler.inverse_transform(
    baseline_scaled.reshape(-1, 1)
).flatten()


baseline_mae = mean_absolute_error(
    actual_values,
    baseline_predictions
)

baseline_mse = mean_squared_error(
    actual_values,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    baseline_mse
)

baseline_r2 = r2_score(
    actual_values,
    baseline_predictions
)


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print("\n==============================")
print("LSTM MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


print("\n==============================")
print("NAIVE BASELINE PERFORMANCE")
print("==============================")

print(f"MAE  : {baseline_mae:.4f}")
print(f"MSE  : {baseline_mse:.4f}")
print(f"RMSE : {baseline_rmse:.4f}")
print(f"R²   : {baseline_r2:.4f}")


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)


# ---------------------------------------------------------
# CREATE RESULTS DIRECTORY
# ---------------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)


# ---------------------------------------------------------
# TRAFFIC PREDICTION GRAPH
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    actual_values,
    label="Actual Traffic"
)

plt.plot(
    predictions,
    label="LSTM Prediction"
)

plt.xlabel(
    "Time Step"
)

plt.ylabel(
    "Vehicle Count"
)

plt.title(
    "Actual vs Predicted Traffic"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/traffic_prediction.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# TRAINING / VALIDATION LOSS GRAPH
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_loss.png",
    dpi=300
)

plt.show()