# 📊 Data Format Documentation

## Required Parquet Files

The application requires two Parquet files located in the `app/` directory.

---

## 📁 `historical_data.parquet`

**Purpose:** Contains actual historical export data

```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┐
│  timestamp  │   Country   │  Branches   │   y_true    │ prediction_type │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤
│ 2015-01-01  │ Netherlands │ FISH TOTAL  │   15234.50  │      null       │
│ 2015-02-01  │ Netherlands │ FISH TOTAL  │   16789.25  │      null       │
│ 2015-03-01  │ Netherlands │ FISH TOTAL  │   14567.80  │      null       │
│     ...     │     ...     │     ...     │     ...     │      ...        │
│ 2024-12-01  │ Spain       │ AL TOTAL    │  125678.00  │      null       │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────────┘

datetime64[ns]    string        string       float64         object
```

### Schema

| Column | Data Type | Required | Description |
|--------|-----------|----------|-------------|
| `timestamp` | `datetime64[ns]` | ✅ Yes | Monthly timestamp (first day of month) |
| `Country` | `string` | ✅ Yes | Export destination country name |
| `Branches` | `string` | ✅ Yes | Industry sector/product category |
| `y_true` | `float64` | ✅ Yes | Actual export value in millions ISK |
| `prediction_type` | `object` | ⚠️ Optional | Should be `null` or empty for historical data |

---

## 📁 `all_predictions.parquet`

**Purpose:** Contains model predictions for validation and future periods

```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┐
│  timestamp  │   Country   │  Branches   │   y_pred    │   y_true    │ prediction_type │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤
│ 2023-01-01  │ Netherlands │ FISH TOTAL  │   15500.00  │   15234.50  │   Validation    │
│ 2023-02-01  │ Netherlands │ FISH TOTAL  │   16900.00  │   16789.25  │   Validation    │
│     ...     │     ...     │     ...     │     ...     │     ...     │      ...        │
│ 2025-01-01  │ Netherlands │ FISH TOTAL  │   18500.00  │     NaN     │     Future      │
│ 2025-02-01  │ Netherlands │ FISH TOTAL  │   19200.00  │     NaN     │     Future      │
│     ...     │     ...     │     ...     │     ...     │     ...     │      ...        │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────────┘

datetime64[ns]    string        string       float64       float64         string
```

### Schema

| Column | Data Type | Required | Description |
|--------|-----------|----------|-------------|
| `timestamp` | `datetime64[ns]` | ✅ Yes | Monthly timestamp (first day of month) |
| `Country` | `string` | ✅ Yes | Export destination country name |
| `Branches` | `string` | ✅ Yes | Industry sector/product category |
| `y_pred` | `float64` | ✅ Yes | Predicted export value in millions ISK |
| `y_true` | `float64` | ⚠️ Conditional | Actual value (required for "Validation", `NaN` for "Future") |
| `prediction_type` | `string` | ✅ Yes | Either `"Validation"` or `"Future"` |

---

## 🔄 Data Processing Flow

```
┌──────────────────────┐      ┌──────────────────────┐
│ historical_data.     │      │ all_predictions.     │
│    parquet           │      │    parquet           │
└──────────┬───────────┘      └──────────┬───────────┘
           │                             │
           └──────────┬──────────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  pd.concat()        │
            │  COMBINED_DATA      │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Add computed fields:│
            │ - year              │
            │ - value = y_true    │
            │   .fillna(y_pred)   │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Dashboard Display  │
            └─────────────────────┘
```

---

## 📋 Important Notes

### Currency Units
- **All monetary values are in millions of Icelandic Króna (ISK)**
- Example: `15234.50` = 15,234.5 million ISK = 15.2345 billion ISK

### Date Format
- Timestamps should be the **first day of each month**
- Format: `YYYY-MM-01` (e.g., `2024-01-01`, `2024-02-01`)
- Frequency: Monthly (`freq='MS'` in pandas)

### Missing Values
- `y_true = NaN` in predictions → Future forecast (no known outcome yet)
- `y_true = value` in predictions → Validation period (can compare prediction vs actual)

### Prediction Types
- **`"Validation"`**: Historical predictions where we have actual outcomes
  - Used to assess model accuracy
  - Both `y_pred` and `y_true` should have values
- **`"Future"`**: Forward-looking forecasts
  - `y_true` should be `NaN`
  - Only `y_pred` has values

---

## 🐍 Python Example: Generating the Files

```python
import pandas as pd
import numpy as np

# Example: Historical Data
historical_df = pd.DataFrame({
    'timestamp': pd.date_range('2015-01-01', '2024-12-01', freq='MS'),
    'Country': 'Netherlands',
    'Branches': 'FISH TOTAL',
    'y_true': np.random.uniform(10000, 20000, 120),  # Replace with actual data
    'prediction_type': None
})
historical_df.to_parquet('app/historical_data.parquet', index=False)

# Example: Predictions (Validation + Future)
validation_period = pd.date_range('2023-01-01', '2024-12-01', freq='MS')
future_period = pd.date_range('2025-01-01', '2026-12-01', freq='MS')

validation_df = pd.DataFrame({
    'timestamp': validation_period,
    'Country': 'Netherlands',
    'Branches': 'FISH TOTAL',
    'y_pred': np.random.uniform(10000, 20000, len(validation_period)),
    'y_true': np.random.uniform(10000, 20000, len(validation_period)),
    'prediction_type': 'Validation'
})

future_df = pd.DataFrame({
    'timestamp': future_period,
    'Country': 'Netherlands',
    'Branches': 'FISH TOTAL',
    'y_pred': np.random.uniform(10000, 20000, len(future_period)),
    'y_true': np.nan,
    'prediction_type': 'Future'
})

predictions_df = pd.concat([validation_df, future_df], ignore_index=True)
predictions_df.to_parquet('app/all_predictions.parquet', index=False)
```

---

## ✅ Validation Checklist

Before deploying new data files, verify:

- [ ] Both files are in Parquet format (`.parquet` extension)
- [ ] Both files are located in the `app/` directory
- [ ] All required columns are present with correct names
- [ ] Data types match the specification
- [ ] Timestamps are first day of month (`YYYY-MM-01`)
- [ ] No duplicate rows (unique combinations of timestamp, Country, Branches)
- [ ] All monetary values are in millions ISK
- [ ] Validation predictions have both `y_pred` and `y_true`
- [ ] Future predictions have `y_true = NaN`
- [ ] `prediction_type` is either `"Validation"` or `"Future"`

---

## 🔍 Quick Data Inspection

```python
import pandas as pd

# Load and inspect
hist = pd.read_parquet('app/historical_data.parquet')
pred = pd.read_parquet('app/all_predictions.parquet')

print("Historical Data:")
print(hist.info())
print(hist.head())

print("\nPredictions:")
print(pred.info())
print(pred.head())
print(pred['prediction_type'].value_counts())
```
