
### Overview

This project provides a **secured forecasting API** that:

- Accepts CSV files containing time-series sales data.
- Cleans and prepares the data for forecasting.
- Runs a **forecasting pipeline** (`forecast_pipeline_debug`) and returns forecast results as a downloadable CSV file.
- Implements **token-based access control**, with per-user credit limits.
- Allows users to generate or refresh tokens through a separate endpoint.

The system ensures **data integrity**, **controlled API access**, and **traceable user activity** with per-user result storage.

## System Architecture


FastAPI Application

│

├── main.py                      # Core API with /forecast endpoint

│   ├── Data Preparation (_prepare_df)

│   ├── Token Validation (_require_token, _decrement_token)

│   ├── Forecast Execution (forecast_pipeline_debug)

│   └── Result Persistence (/user_data/`<username>`/)

│

├── accesstoken.py               # Token management routes (/generate-token)

│   ├── Generate / Refresh Tokens

│   ├── Validate Ownership

│   ├── Persist tokens in CSV

│

└── Token/

└── user_token.csv           # Persistent token store (username, token, credits)


## Core Concepts


### 1. Forecast Endpoint (`/forecast`)

Runs the full forecasting pipeline for user-uploaded data. Access requires a valid token.

**Workflow:**

1. Validate user token.
2. Load uploaded CSV.
3. Clean data.
4. Run forecast model.
5. Deduct token credits per unique key.
6. Save output under `/user_data/<username>/`.
7. Return forecast CSV.


## Data Preparation

### `_prepare_df(df: pd.DataFrame, params: Dict)`

| Step | Description                                                             |
| ---- | ----------------------------------------------------------------------- |
| 1    | Converts all column names to lowercase.                                 |
| 2    | Parses date column into datetime objects.                               |
| 3    | Builds `key` column using concatenation of `key_components`.        |
| 4    | Ensures target column is numeric.                                       |
| 5    | Handles `seasonal_col`: fills with default (Y/N) and normalizes case. |
| 6    | Aggregates data to monthly level.                                       |
| 7    | Adds or computes `hist_range` column.                                 |
| 8    | Returns cleaned DataFrame ready for forecasting.                        |

## Token Management System

Tokens are used to restrict API usage and assign credit-based access per user.

### Token CSV

## Token API 

Token/user_token.csv

| username | token   | counter |
| -------- | ------- | ------- |
| akshat   | abcd123 | 100     |


### `_require_token(token: str)`

Validates token existence and available credits.

### `_decrement_token(token: str, amount: int)`

Reduces token counter after forecast execution.

### `_create_new_credentials()`

Utility for generating username-token pairs.

## Forecast API

**Endpoint:** `POST /forecast`

### Form Parameters

| Name                | Type       | Required                | Description               |
| ------------------- | ---------- | ----------------------- | ------------------------- |
| file                | UploadFile | ✅                      | CSV input file            |
| csv_path            | str        | ❌                      | Alternative to upload     |
| date_col            | str        | ✅                      | Date column name          |
| target_col          | str        | ✅                      | Sales/target column       |
| key_col             | str        | ✅                      | Unique time series ID     |
| key_components      | str        | ✅ (if key_col missing) | Columns for key creation  |
| seasonal_col        | str        | ❌                      | Seasonality flag          |
| seasonal_default    | str        | ❌                      | Default if missing        |
| validation_cutoff   | str        | ✅                      | Validation split date     |
| test_cutoff         | str        | ✅                      | Test split date           |
| forecast_cutoff     | str        | ✅                      | Forecast end date         |
| forecasting_horizon | int        | ❌                      | Default = 3               |
| token               | str        | ✅                      | User authentication token |
| username            | str        | ❌                      | Auto-filled from token    |

### Example Request


```bash
curl -X POST "http://127.0.0.1:8000/forecast" \\
  -F "file=@C:\\path\\data.csv" \\
  -F "date_col=sales_date" \\
  -F "target_col=sales" \\
  -F "key_components=business_type x store_type x region x subcat x product_code" \\
  -F "validation_cutoff=2024-04-30" \\
  -F "test_cutoff=2024-07-31" \\
  -F "forecast_cutoff=2024-12-31" \\
  -F "token=<your_token>" \\
  -o "forecast_results.csv"
```


## Token API

**Endpoint:** `POST /generate-token`

### Form Parameters

| Name     | Type | Required | Description                      |
| -------- | ---- | -------- | -------------------------------- |
| username | str  | ✅       | User ID                          |
| token    | str  | ❌       | Custom token                     |
| counter  | int  | ❌       | Starting credits (default = 100) |

**Response:**

{
  "username": "akshat",
  "token": "S0YHfdd123vA9QkL...",
  "counter": 100,
  "status": "created",
  "message": "token created for username; initial counter=100"
}

## Server Structure

project_root/
├── main.py
├── accesstoken.py
├── forecasting_pipeline.py
├── Token/
│   └── user_token.csv
└── user_data/
    ├── akshat/
    │   └── forecast_akshat_20251101_103212.csv

## Error Handling

| Error Type       | Code | Example                                |
| ---------------- | ---- | -------------------------------------- |
| Invalid Token    | 401  | "Invalid token"                        |
| Expired Token    | 401  | "Token expired (no remaining credits)" |
| Missing File     | 400  | "Please supply an uploaded file"       |
| Invalid Columns  | 400  | "Date column not found"                |
| Forecast Failure | 500  | "Forecast pipeline failed"             |

## Dependencies

fastapi
uvicorn
pandas
numpy
pmdarima
python-dateutil
typing-extensions

## Running the App

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

## Summary

| Component       | File                       | Purpose                            |
| --------------- | -------------------------- | ---------------------------------- |
| Forecast API    | main.py                    | CSV upload, data prep, forecasting |
| Token API       | accesstoken.py             | Token creation & management        |
| Token Store     | Token/user_token.csv       | Token persistence                  |
| Forecast Engine | forecasting_pipeline.py    | ML-based forecasting logic         |
| User Storage    | /user_data/`<username>`/ | User-specific results              |
|                 |                            |                                    |



### **Author:** Akshat Verma

**Framework:** FastAPI
**Language:** Python 3.12+
**Modules Used:** `fastapi`, `pandas`, `secrets`, `datetime`, `io`, `os`, `logging`, `re`, `numpy`
