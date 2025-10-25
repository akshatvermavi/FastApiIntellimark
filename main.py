import os
import io
import logging
import re
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd

# Import the pipeline function and helper from your file
from forecasting_pipeline import forecast_pipeline_debug, add_hist_range

app = FastAPI(title="Monthly Forecast API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default parameters used for calling the pipeline
DEFAULT_PARAMS = {
    "date_col": "date",
    "target_col": "unitssold",
    "key_col": "key",
    "seasonal_col": "seasonal",
    "hist_range_col": "hist_range",
    "key_components": None,  # optional: comma or 'x'-separated list of columns to build key
    "seasonal_default": "NA",  # user can override; will be normalized to Y/N internally
}


def _prepare_df(df: pd.DataFrame, params: Dict):
    """
    Ensure minimal required columns exist for the forecast pipeline:
    - date_col exists and is parsed
    - create 'key' column (if not present) by concatenating dimensions
    - ensure seasonal_col is present (default 'N')
    - compute hist_range via add_hist_range (if missing)
    - remove duplicates and aggregate monthly values
    Returns cleaned DataFrame.
    """
    date_col = params["date_col"]
    key_col = params["key_col"]
    seasonal_col = params["seasonal_col"]
    hist_range_col = params["hist_range_col"]

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    # print("✅ df completed")
    # df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df4.csv", index=False)
    # Parse date
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in uploaded data.")
    # df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
    # df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    # print("✅ df completed")
    # df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df5.csv", index=False)
    if df[date_col].isna().all():
        raise ValueError(f"Could not parse dates in column '{date_col}'. Please ensure a valid date format.")

    # If key column not in df, create key by concatenating provided components or fallbacks
    if key_col not in df.columns:
        key_components = params.get("key_components")
        if not key_components:
            raise ValueError("Key column not found in the data. Provide 'key_components' to build it or include 'key_col' in the CSV.")
        parts = [p.strip().lower() for p in re.split(r"x|,|\+|\||;", key_components) if p.strip()]
        missing = [p for p in parts if p not in df.columns]
        if missing:
            raise ValueError(f"Missing key component columns: {missing}. Available: {list(df.columns)}")
        df[key_col] = df[parts].astype(str).agg("_".join, axis=1)

        # candidates: list[str] = []
        # if key_components:
        #     # split on 'x' or commas and normalize case/spaces
        #     parts = [p.strip().lower() for p in re.split(r"x|,|\+|\||;", key_components) if p.strip()]
        #     missing = [p for p in parts if p not in df.columns]
        #     if missing:
        #         raise ValueError(f"Missing key component columns: {missing}. Available: {list(df.columns)}")
        #     candidates = parts
        # else:  
        #     candidates = [c for c in ["channel", "chain", "depot", "subcat", "sku"] if c in df.columns]
        # if not candidates:
        #     df[key_col] = "GLOBAL"
        # else:
        #     df[key_col] = df[candidates].astype(str).agg("_".join, axis=1)

    # Ensure target_col exists and numeric
    # print("✅ df completed")
    # df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df6.csv", index=False)
    target_col = params["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in uploaded data.")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Seasonal column: if missing, create with provided default; normalize to Y/N
    seasonal_default = str(params.get("seasonal_default", "NA")).upper()
    if seasonal_col not in df.columns:
        df[seasonal_col] = seasonal_default
    else:
        df[seasonal_col] = df[seasonal_col].fillna(seasonal_default)
    # normalize values so pipeline rules work (NA -> N)
    df[seasonal_col] = df[seasonal_col].astype(str).str.upper().str.strip()
    df[seasonal_col] = df[seasonal_col].apply(lambda v: "Y" if v.startswith("Y") else ("N" if v.startswith("N") else "N"))
    # print("✅ df completed")        
    # df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df7.csv", index=False)
    # Convert dates to month start (pipeline expects monthly frequency)
    # NOTE: Period.to_timestamp expects a period frequency (e.g., 'M'), not a date-offset like 'MS'.
    # Using 'MS' here raises: "MS is not supported as period frequency".
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
    # print("✅ df completed")
    # df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df8.csv", index=False)
    # Aggregate duplicate rows but RETAIN seasonal by grouping with it
    # This ensures downstream pipeline has 'seasonal' available per key
    if seasonal_col not in df.columns:
        df[seasonal_col] = "N"
    df = (
        df.groupby([key_col, seasonal_col, date_col], as_index=False)[target_col]
          .sum()
    )
    # print("✅ df completed")
    # df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df9.csv", index=False)
    # Hist_range column: compute using helper (per key) and map back
    if hist_range_col not in df.columns:
        temp = df.rename(columns={date_col: "date", key_col: "key", target_col: "actual_value"})
        temp = add_hist_range(temp, key_col='key', date_col='date')
        if 'hist_range' in temp.columns:
            hist_map = (
                temp[["key", "hist_range"]]
                .drop_duplicates()
                .set_index("key")["hist_range"].to_dict()
            )
            df[hist_range_col] = df[key_col].map(hist_map)
        df[hist_range_col] = df.get(hist_range_col, "<6").fillna("<6")
    else:
        df[hist_range_col] = df[hist_range_col].fillna("<6")

    # Final clean-up
    #df = df.drop_duplicates().reset_index(drop=True)
    # print("✅ df completed")
    # df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df10.csv", index=False)
    return df
        # Hist_range column: compute if missing
    # if hist_range_col not in df.columns:
    #     temp = df.rename(columns={date_col: "date", key_col: "key", target_col: "actual_value"})
    #     temp = add_hist_range(temp, key_col='key', date_col='date')
    #     hist_map = temp[['key','hist_range']].drop_duplicates().set_index('key')['hist_range'].to_dict()
    #     df[hist_range_col] = df[key_col].map(hist_map)
    # df[hist_range_col] = df[hist_range_col].fillna("<6")

    # # Ensure seasonal column exists
    # if seasonal_col not in df.columns:
    #     df[seasonal_col] = "N"
    # else:
    #     df[seasonal_col] = df[seasonal_col].fillna("N")

    # # Drop duplicates at the end
    # df = df.drop_duplicates()

    # logger.info(f"✅ Added/verified seasonal and hist_range columns.")
    # return df



@app.post("/forecast")
async def run_forecast(
    csv_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    date_col: str = Form(DEFAULT_PARAMS["date_col"]),
    target_col: str = Form(DEFAULT_PARAMS["target_col"]),
    key_col: str = Form(DEFAULT_PARAMS["key_col"]),
    seasonal_col: str = Form(DEFAULT_PARAMS["seasonal_col"]),
    hist_range_col: str = Form(DEFAULT_PARAMS["hist_range_col"]),
    validation_cutoff: str = Form(...),
    test_cutoff: str = Form(...),
    forecast_cutoff: str = Form(...),
    forecasting_horizon: int = Form(3),
    debug_keys: Optional[str] = Form(None),
    key_components: Optional[str] = Form(DEFAULT_PARAMS["key_components"]),
    seasonal_default: str = Form(DEFAULT_PARAMS["seasonal_default"])  # e.g., Y/N/NA
):
# async def run_forecast(
#     csv_path: Optional[str] = Form(None),
#     file: Optional[UploadFile] = File(None),
#     date_col: str = Form(DEFAULT_PARAMS["date_col"]),
#     target_col: str = Form(DEFAULT_PARAMS["target_col"]),
#     key_col: str = Form(DEFAULT_PARAMS["key_col"]),
#     seasonal_col: str = Form(DEFAULT_PARAMS["seasonal_col"]),
#     hist_range_col: str = Form(DEFAULT_PARAMS["hist_range_col"]),
#     validation_cutoff: str = Form(...),
#     test_cutoff: str = Form(...),
#     forecast_cutoff: str = Form(...),
#     forecasting_horizon: int = Form(3),
#     debug_keys: Optional[str] = Form(None),
#     key_components: Optional[str] = Form(DEFAULT_PARAMS["key_components"])
# ):
    """
    Run forecast pipeline.
    - Either upload a CSV file or provide csv_path.
    - Required: validation_cutoff, test_cutoff, forecast_cutoff.
    - Optional: forecasting_horizon (default 3)
    """
    # Load dataframe
    #df.columns = [c.lower() for c in df.columns]
    if file is not None:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded CSV: {e}")
    elif csv_path:
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=400, detail=f"csv_path not found: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise HTTPException(status_code=400, detail="Please supply an uploaded file or a csv_path.")

    # Prepare params dict
    params = {
        "date_col": date_col,
        "target_col": target_col,
        "key_col": key_col,
        "seasonal_col": seasonal_col,
        "hist_range_col": hist_range_col,
        "key_components": key_components,
        "seasonal_default": seasonal_default,
    }

    # Prepare/clean df
    try:
        df_prepared = _prepare_df(df, params)
        logger.info(f"Prepared DataFrame shape: {df_prepared.shape}")
        logger.info(f"Columns: {list(df_prepared.columns)}")
        # ✅ Add this block here — seasonal override logic
        # if seasonal_flag:
        #     logger.info(f"Overriding seasonal flag with user input: {seasonal_flag}")
        #     df_prepared["seasonal"] = seasonal_flag.upper()
        # else:
        #     # If user didn't pass seasonal_flag, ensure seasonal defaults to 'N'
        #     if "seasonal" not in df_prepared.columns:
        #         df_prepared["seasonal"] = "N"
        #     df_prepared["seasonal"] = df_prepared["seasonal"].fillna("N")
        # Detect if seasonal flag exists in CSV
        # --- Handle seasonal flag robustly ---
  # === Handle Seasonal Column Logic ===
        possible_seasonal_cols = ["seasonal", "seasonal_flag", "season_flag", "season"]
        existing = [c for c in possible_seasonal_cols if c in df.columns]

        if existing:
        # use existing column
            df.rename(columns={existing[0]: seasonal_col}, inplace=True)
            df[seasonal_col] = df[seasonal_col].fillna("N").astype(str).str.upper()
        else:
        # No seasonal column in CSV → use user override flag or set to N
            if seasonal_flag:
                df[seasonal_col] = seasonal_flag.upper()
            else:
                df[seasonal_col] = "N"



    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Handle debug_keys
    debug_keys_list = None
    if debug_keys:
        debug_keys_list = [k.strip() for k in debug_keys.split(",") if k.strip()]

    # Run pipeline
    try:
        result = forecast_pipeline_debug(
            df=df_prepared,
            parameters=params,
            validation_cutoff=validation_cutoff,
            test_cutoff=test_cutoff,
            forecast_cutoff=forecast_cutoff,
            forecasting_horizon=forecasting_horizon,
            debug_keys=debug_keys_list
        )
        # If it returns a tuple, take the first element as the DataFrame
        if isinstance(result, tuple):
            result_df = result[0]
        else:
            result_df = result

    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=f"Forecast pipeline failed: {e}")

    # Return CSV stream
    out_buf = io.StringIO()
    result_df.to_csv(out_buf, index=False)
    out_buf.seek(0)
    csv_bytes = out_buf.getvalue().encode('utf-8')

    return StreamingResponse(io.BytesIO(csv_bytes),
                             media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=forecast_results.csv"})
