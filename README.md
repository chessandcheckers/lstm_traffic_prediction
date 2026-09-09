# LSTM Traffic Prediction

> **Status:** Actively rebuilding · Junction 1 pipeline complete · Baseline comparisons in progress

A time-series forecasting project using a Long Short-Term Memory (LSTM) neural network to predict hourly urban traffic volume from historical traffic patterns.

This repository is a revisit and rebuild of a model I originally developed as a first-year hackathon project.

---

## Project Background

The original project was built during a hackathon by a team of four students: three first-year students, including me, and one second-year student.

Our approach was to explore multiple machine learning and deep learning models for traffic-flow prediction. Each team member worked on a different model, with the intention of comparing different approaches to the same problem.

My contribution was the **LSTM-based traffic prediction model**.

At the time, the three first-year members primarily worked on developing their individual models and did not integrate them into a frontend. Our second-year teammate developed a more complete implementation with a functioning user interface, which became the version used for the final hackathon presentation.

My original LSTM implementation was therefore primarily a model prototype rather than a complete application.

## Why Revisit It?

More than a year later, I returned to the project to rebuild my original contribution with a better understanding of Python, machine learning workflows, time-series data, model evaluation, and software project structure.

Rather than presenting the old hackathon code as a finished project, this repository documents the process of turning that early prototype into a reproducible and properly evaluated forecasting model.

The current version includes:

- chronological train/test splitting for time-series data
- feature engineering from timestamps
- data normalization using `MinMaxScaler`
- sequence generation for LSTM input
- stacked LSTM layers with dropout
- early stopping based on validation loss
- evaluation on unseen test data using MAE, MSE, RMSE, and R²
- comparison against a naive persistence baseline
- visualization of predictions and training behaviour
- model serialization for later inference

---

## Dataset

The current implementation uses the **Traffic Prediction Dataset** published by Federico Soriano on Kaggle.

The dataset contains hourly traffic observations from multiple junctions:

| Feature | Description |
|---|---|
| `DateTime` | Timestamp of the traffic observation |
| `Junction` | Junction identifier |
| `Vehicles` | Number of vehicles observed |
| `ID` | Unique observation identifier |

The dataset is not included in this repository. Place the downloaded file at:

```text
data/traffic.csv
```

---

## Current Model

The current implementation focuses on **Junction 1** to build and validate a clear single-location forecasting pipeline before extending to multiple junctions.

For every prediction, the model receives the previous **24 hours** of observations.

Input features:

```text
Vehicles · Hour · DayOfWeek · Month
```

Target:

```text
Vehicle count for the next hour
```

LSTM input shape: `(samples, 24 timesteps, 4 features)`

### Architecture

```text
24-hour input sequence
         │
         ▼
      LSTM (64)
         │
         ▼
    Dropout (0.2)
         │
         ▼
      LSTM (32)
         │
         ▼
    Dropout (0.2)
         │
         ▼
    Dense (16, ReLU)
         │
         ▼
       Dense (1)
         │
         ▼
Next-hour vehicle prediction
```

Trained using the Adam optimizer with Mean Squared Error loss. Early stopping monitors validation loss and restores the best-performing weights.

---

## Results

The current model achieves the following on unseen test observations for Junction 1:

```text
MAE  : 9.2037
MSE  : 130.0253
RMSE : 11.4029
R²   : 0.7678
```

The model captures recurring temporal patterns reasonably well. The prediction graph shows a tendency to smooth out sudden traffic spikes — a known limitation of sequence models on high-variance events.

![Actual vs Predicted Traffic](results/traffic_prediction.png)

> These are current results, not final benchmarks. The model is being improved and evaluated against a naive baseline before drawing conclusions about forecasting performance.

---

## Project Structure

```text
lstm_traffic_prediction/
│
├── data/
│   └── traffic.csv
│
├── models/
│   └── traffic_lstm.keras
│
├── results/
│   ├── traffic_prediction.png
│   └── training_loss.png
│
├── src/
│   └── train_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Running the Project

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `traffic.csv` inside `data/` and run:

```bash
python src/train_model.py
```

The script trains the model, evaluates predictions, saves the trained model, and generates result visualizations.

---

## Current Limitations

- trains on a single junction only
- vehicle count used as a proxy for traffic volume rather than physical density
- no contextual features: weather, accidents, holidays, road conditions, or events
- sudden traffic spikes are harder to predict accurately
- baseline comparisons and further model experiments still in progress

## Planned Improvements

- evaluate LSTM against simple forecasting baselines
- extend pipeline to multiple junctions
- improved temporal feature engineering
- experiments with different sequence lengths and architectures
- hyperparameter tuning
- richer evaluation and visualizations
- separate inference pipeline
- possible integration with a simple API or user interface

---

## Tech Stack

`Python` · `TensorFlow / Keras` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib`

---

## Note

This repository represents my **individual LSTM contribution and its subsequent rebuild**, not the complete application presented by the original hackathon team.

The purpose of revisiting this project is not to make the original prototype appear more complete than it was — it's to document how an early first-year experiment improves as technical understanding develops.