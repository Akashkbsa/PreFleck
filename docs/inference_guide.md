# Production Inference Guide

To make predictions on **brand new data streams**, you cannot simply feed a single row of sensor readings into the model. Because the V2 model relies heavily on temporal features (like 24-hour rolling averages, derivatives, and lags), it requires a **historical buffer**.

Here is the exact framework to implement live predictions.

---

## 1. The Historical Buffer Requirement
Whenever new sensor data arrives for a `machineID`, you must append it to a temporary historical database. To calculate all V2 features correctly, you must retain the last **72 hours** (3 days) of data for that machine.
- `volt`, `rotate`, `pressure`, `vibration`
- `errors` (if any occurred)
- `maintenance` (if any occurred)

## 2. The Inference Workflow
When you want to predict the 48-hour failure probability for `Machine_1` at `Time T`:

1. **Query Buffer**: Fetch data from `T-72h` to `T` for `Machine_1`.
2. **Preprocess**: Run the standard `preprocessing.py` logic (cleaning / merging).
3. **Feature Engineer**: Run the `feature_engineering_v2.py` logic on this 72-hour window. This automatically generates the correct rolling means, EMAs, and derivatives for the final row (`Time T`).
4. **Predict**: Extract only the very last row (`Time T`) and pass it to `model.predict_proba()`.

---

## 3. Save the Trained Model
Before running inference, we must export your trained XGBoost model from `model_training_v2.py`.

```python
# During training
model.save_model("xgboost_v2_48h.json")
```

## 4. Example Live Data Input Format
Your incoming data payload should look like this (or be structured in a database):
```json
{
    "datetime": "2026-10-20 08:00:00",
    "machineID": 1,
    "volt": 172.1,
    "rotate": 450.2,
    "pressure": 105.1,
    "vibration": 42.1,
    "errors": ["error1"], 
    "maintenance": []
}
```
*Note: A complete reference implementation is available in `src/inference.py`.*
