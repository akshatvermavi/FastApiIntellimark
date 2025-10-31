import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import optuna
from typing import Dict, List, Optional

# Get a logger instance for the module
logger = logging.getLogger(__name__)


MODEL_RULES = {
    # Seasonal Time Series
    ("Y", "24+"): ["HoltWinters", "SARIMA", "Prophet", "TBF","LY_trend", "RF_default", "XGB_default", "LGBM_default"],
    ("Y", "13–24"): ["HoltWinters", "SARIMA", "Prophet", "TBF","LY_trend", "RF_default", "XGB_default", "LGBM_default"],
    ("Y", "7–12"): [ "L3M", "L6M", "HoltWinters",  "SARIMA", "Prophet", "Naive"],
    ("Y", "<6"): ["L3M", "Naive"],
    # Non-Seasonal Time Series
    ("N", "24+"): ["TBF", "L3M", "L6M", "Prophet", "LY_trend", "RF_default", "XGB_default", "LGBM_default"],
    ("N", "13–24"): [ "TBF", "L3M", "L6M", "LY_trend", "RF_default", "XGB_default", "LGBM_default"],
    ("N", "7–12"): ["L3M", "L6M", "Naive"],
    ("N", "<6"): ["L3M", "Naive"]
}

def calculate_bias_adjustment(val_actual: pd.Series, val_forecast: pd.Series, future_forecast: pd.Series):
    """
    Calculates a bias adjustment factor from a validation set and applies it to a future forecast set.
    """
    # Ensure there's valid data to learn from and that forecasts are not zero
    if val_actual.empty or val_forecast.empty or future_forecast.empty or val_actual.sum() == 0 or val_forecast.sum() == 0:
        return future_forecast # Return original forecast if adjustment is not possible

    # Calculate bias ratio for each validation point, handling potential division by zero
    bias = np.divide(
        val_actual.values - val_forecast.values,
        val_actual.values,
        out=np.zeros_like(val_actual.values, dtype=float),
        where=val_actual.values != 0
    )

    # Check for consistent bias (all positive or all negative, ignoring zeros)
    if (np.all(bias[bias != 0] > 0)) or (np.all(bias[bias != 0] < 0)):
        # Calculate dampened adjustment factor
        adj_factor = (((val_actual.sum() / val_forecast.sum()) - 1) / 2) + 1
    else:
        adj_factor = 1.0 # No consistent bias, no adjustment
    
    return future_forecast * adj_factor


def calculate_tbf_trend_ets(data, key_col, date_col, target_col, train_cutoff, ma_order=9):
    """
    Calculate trend using ETS model with moving average (matches original TBF implementation)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Filter training data
    train_data = data[data[date_col] <= train_cutoff].copy()
    
    if len(train_data) < ma_order:
        # Fallback to simple mean for short series
        return train_data[target_col].mean()
    
    # ✅ FIXED: Calculate MA-based trend like original
    train_data = train_data.sort_values(date_col).set_index(date_col)
    train_data = train_data.asfreq('MS')  # Ensure monthly frequency like original
    
    # Calculate MA-based trend
    trend_series = train_data[target_col].rolling(window=ma_order, min_periods=1).mean()
    
    try:
        # ✅ FIXED: Fit ETS model like original
        model = ExponentialSmoothing(
            trend_series,
            trend='add',  # Same as original
            seasonal=None,  # Same as original
            initialization_method='estimated'  # Same as original
        ).fit()
        
        # ✅ FIXED: Return the last fitted value (not forecast)
        return model.fittedvalues.iloc[-1]
        
    except Exception as e:
        logger.warning(f"ETS model failed, using MA trend: {e}")
        return trend_series.iloc[-1]
    


def calculate_tbf_seasonality_robust(data, key_col, date_col, target_col, train_cutoff):
    """
    Calculate robust seasonal ratios using TBF methodology (matches original implementation)
    """
    train_data = data[data[date_col] <= train_cutoff].copy()
    train_data['month'] = train_data[date_col].dt.month
    
    # ✅ FIXED: Filter positive values only (like original)
    train_data_positive = train_data[train_data[target_col] > 0]
    
    if train_data_positive.empty:
        # Return equal distribution if no positive data
        return {month: 1/12 for month in range(1, 13)}
    
    # ✅ STEP 1: Calculate average monthly sales (like original)
    avg_monthly_sales = train_data_positive.groupby('month')[target_col].mean()
    
    # ✅ STEP 2: Calculate "typical yearly total" (like original)
    typical_yearly_total = avg_monthly_sales.sum()
    
    # ✅ STEP 3: Calculate robust yearly ratios (like original)
    if typical_yearly_total > 0:
        monthly_ratios = (avg_monthly_sales / typical_yearly_total).to_dict()
    else:
        monthly_ratios = {month: 1/12 for month in range(1, 13)}
    
    # ✅ STEP 4: Fill missing months with default ratio
    for month in range(1, 13):
        if month not in monthly_ratios:
            monthly_ratios[month] = 1/12
    
    return monthly_ratios



def apply_tbf_upper_lower_bounds(forecast_values, historical_data, rolling_window=6):
    """
    Apply upper/lower bounds using TBF methodology (matches original implementation)
    """
    if len(historical_data) < rolling_window:
        return forecast_values  # No bounding if insufficient data
    
    # ✅ FIXED: Use the exact same logic as original build_monthly_upper_lower_band
    recent_data = historical_data.iloc[-rolling_window:]
    
    # Calculate rolling bounds (like original)
    rolling_max = recent_data.rolling(window=rolling_window, center=True).max().iloc[-1]
    rolling_min = recent_data.rolling(window=rolling_window, center=True).min().iloc[-1]
    
    # Apply thresholds (like original)
    min_actual_threshold = max(recent_data.min(), 0)
    max_actual_threshold = recent_data.max()
    
    upper_bound = max(rolling_max, max_actual_threshold) if not pd.isna(rolling_max) else max_actual_threshold
    lower_bound = max(rolling_min, min_actual_threshold) if not pd.isna(rolling_min) else min_actual_threshold
    
    # Apply bounds with same logic as original "capped_pred"
    bounded_forecast = []
    for val in forecast_values:
        if val > upper_bound:
            bounded_forecast.append(upper_bound)
        elif val < lower_bound:
            bounded_forecast.append(lower_bound)
        else:
            bounded_forecast.append(val)
    
    return np.array(bounded_forecast)


def run_TBF(series, horizon=3, freq="MS", train_cutoff=None, 
            key_name=None, full_df=None, key_col=None, target_col=None, 
            date_col=None, ma_order=9, rolling_window=6, **kwargs):
    """
    Trend-Based Forecasting model for Monthly Baseline Pipeline
    """
    
    if series.dropna().shape[0] < 3:
        return pd.Series(dtype=float)
    
    # Set default train_cutoff if not provided
    if train_cutoff is None:
        train_cutoff = series.index[-horizon-1] if len(series) > horizon else series.index[0]
    
    last_date = series.index[-1]
    future_dates = pd.date_range(
        last_date + pd.tseries.frequencies.to_offset(freq),
        periods=horizon, 
        freq=freq
    )
    
    try:
        # ✅ FIXED: Prepare data in TBF format with proper structure
        tbf_data = pd.DataFrame({
            date_col or 'date': series.index,
            target_col or 'target': series.values,
            key_col or 'key': key_name or 'default_key'
        })
        
        # ✅ STEP 1: Calculate ETS-based trend using moving average (like original)
        trend_value = calculate_tbf_trend_ets(
            tbf_data, 
            key_col or 'key', 
            date_col or 'date', 
            target_col or 'target', 
            train_cutoff, 
            ma_order
        )
        
        # ✅ STEP 2: Calculate robust seasonality ratios (like original)
        seasonal_ratios = calculate_tbf_seasonality_robust(
            tbf_data, 
            key_col or 'key', 
            date_col or 'date', 
            target_col or 'target', 
            train_cutoff
        )
        
        # ✅ STEP 3: Generate yearly trend predictions
        yearly_trend = trend_value * 12  # Convert monthly trend to yearly
        
        # ✅ STEP 4: Apply seasonality multiplication (like original)
        forecast_values = []
        for future_date in future_dates:
            month = future_date.month
            seasonal_factor = seasonal_ratios.get(month, 1/12)
            
            # TBF prediction: yearly_trend * seasonal_ratio
            forecast_value = yearly_trend * seasonal_factor
            forecast_values.append(max(forecast_value, 0))  # Ensure non-negative
        
        # ✅ STEP 5: Apply upper/lower bounds (like original)
        historical_values = series[series.index <= train_cutoff].dropna()
        if len(historical_values) >= rolling_window:
            bounded_forecasts = apply_tbf_upper_lower_bounds(
                forecast_values, 
                historical_values, 
                rolling_window
            )
        else:
            bounded_forecasts = forecast_values
        
        return pd.Series(bounded_forecasts, index=future_dates)
        
    except Exception as e:
        logger.warning(f"TBF failed for series {key_name}: {e}")
        return pd.Series(np.nan, index=future_dates)
    

def calculate_monthly_ratios(df_monthly, train_end_date, key_col, target_col):
    """
    Calculates the average proportion of sales for each month within a year using only training data.
    This creates monthly seasonality ratios that can be used as features in ML models.
    """
    train_df = df_monthly[df_monthly['date'] <= pd.to_datetime(train_end_date)].copy()
    
    # Ensure necessary date components exist
    if 'year' not in train_df.columns:
        train_df['year'] = train_df['date'].dt.year
    if 'month' not in train_df.columns:
        train_df['month'] = train_df['date'].dt.month
    
    # Calculate annual totals for each key-year combination
    annual_totals = train_df.groupby([key_col, 'year'])[target_col].sum().rename('annual_total_sales').reset_index()
    
    # Merge annual totals back to monthly data
    train_df = pd.merge(train_df, annual_totals, on=[key_col, 'year'], how='left')
    
    # Calculate monthly ratio (proportion of annual sales for each month)
    # Avoid division by zero
    train_df['monthly_ratio'] = np.where(
        train_df['annual_total_sales'] > 0, 
        train_df[target_col] / train_df['annual_total_sales'], 
        0
    )
    
    # Calculate average monthly ratios across all years for each key-month combination
    # This gives us the typical seasonal pattern for each month
    avg_monthly_ratios = train_df.groupby([key_col, 'month'])['monthly_ratio'].mean().rename('avg_monthly_ratio').reset_index()
    
    # Ensure all 12 months are represented for each key (fill missing months with 1/12)
    all_keys = avg_monthly_ratios[key_col].unique()
    all_months = range(1, 13)
    
    # Create complete key-month combinations
    complete_combinations = pd.MultiIndex.from_product([all_keys, all_months], names=[key_col, 'month']).to_frame(index=False)
    
    # Merge and fill missing values
    avg_monthly_ratios = pd.merge(complete_combinations, avg_monthly_ratios, on=[key_col, 'month'], how='left')
    avg_monthly_ratios['avg_monthly_ratio'] = avg_monthly_ratios['avg_monthly_ratio'].fillna(1/12)  # Default to equal distribution
    
    return avg_monthly_ratios

def calculate_monthly_ratios_dynamic(df_monthly, train_end_date, key_col, target_col, key_filter=None):
    """
    Calculates the average proportion of sales for each month within a year using only training data.
    This version is designed to be called dynamically with different cutoff dates.
    """

    # ✅ FIX: Ensure date column exists and convert dates properly
    if 'date' not in df_monthly.columns:
        logger.warning(f"Warning: 'date' column not found in df_monthly. Available columns: {df_monthly.columns.tolist()}")
        return pd.DataFrame(columns=[key_col, 'month', 'avg_monthly_ratio'])
    
    # ✅ FIX: Convert both date column and train_end_date to datetime
    df_monthly = df_monthly.copy()
    df_monthly['date'] = pd.to_datetime(df_monthly['date'])
    train_end_date = pd.to_datetime(train_end_date)

    # Filter by cutoff date first
    train_df = df_monthly[df_monthly['date'] <= pd.to_datetime(train_end_date)].copy()
    
    # Filter by specific keys if provided (for efficiency)
    if key_filter is not None:
        if isinstance(key_filter, (str, int)):
            key_filter = [key_filter]
        train_df = train_df[train_df[key_col].isin(key_filter)]
    
    if train_df.empty:
        return pd.DataFrame(columns=[key_col, 'month', 'avg_monthly_ratio'])
    
    # Ensure necessary date components exist
    if 'year' not in train_df.columns:
        train_df['year'] = train_df['date'].dt.year
    if 'month' not in train_df.columns:
        train_df['month'] = train_df['date'].dt.month
    
    # Calculate annual totals for each key-year combination
    annual_totals = train_df.groupby([key_col, 'year'])[target_col].sum().rename('annual_total_sales').reset_index()
    
    # Merge annual totals back to monthly data
    train_df = pd.merge(train_df, annual_totals, on=[key_col, 'year'], how='left')
    
    # Calculate monthly ratio (proportion of annual sales for each month)
    # Avoid division by zero
    train_df['monthly_ratio'] = np.where(
        train_df['annual_total_sales'] > 0, 
        train_df[target_col] / train_df['annual_total_sales'], 
        0
    )
    
    # Calculate average monthly ratios across all years for each key-month combination
    # This gives us the typical seasonal pattern for each month
    avg_monthly_ratios = train_df.groupby([key_col, 'month'])['monthly_ratio'].mean().rename('avg_monthly_ratio').reset_index()
    
    # Get unique keys from filtered data
    unique_keys = avg_monthly_ratios[key_col].unique()
    all_months = range(1, 13)
    
    # Create complete key-month combinations
    complete_combinations = pd.MultiIndex.from_product([unique_keys, all_months], names=[key_col, 'month']).to_frame(index=False)
    
    # Merge and fill missing values
    avg_monthly_ratios = pd.merge(complete_combinations, avg_monthly_ratios, on=[key_col, 'month'], how='left')
    avg_monthly_ratios['avg_monthly_ratio'] = avg_monthly_ratios['avg_monthly_ratio'].fillna(1/12)  # Default to equal distribution
    
    return avg_monthly_ratios

def get_monthly_ratios_for_cutoff(df, key, cutoff_date, key_col, target_col, date_col):
    """
    Helper function to get monthly ratios for a specific key and cutoff date.
    This should be called within the pipeline for each phase.
    """
    # This function relies on `calculate_monthly_ratios_dynamic`, which is assumed
    # to be in the full script.
    key_df = df[df[key_col] == key].copy()
    key_df[date_col] = pd.to_datetime(key_df[date_col])
    cutoff_date = pd.to_datetime(cutoff_date)
    monthly_ratios = calculate_monthly_ratios_dynamic(
        key_df,
        train_end_date=cutoff_date,
        key_col=key_col,
        target_col=target_col,
        key_filter=key
    )
    return monthly_ratios




def lowess_trend_seasonality_extrapolate(
    df,
    key_col,
    date_col,
    target_col,
    extrapolate_months=3,
    frac=0.3,
    period=12,
    alpha=0.85,          # recency weight: weight = alpha**age_in_years
    robust_stl=True
):
    """
    Per-key LOWESS trend + STL seasonality (additive, non-negative),
    with trend extrapolated and seasonality for future months filled
    using a recency-weighted monthly index.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([key_col, date_col])

    out = []

    for key, sub in df.groupby(key_col, sort=False):
        g = sub.sort_values(date_col).copy()
        # ----- LOWESS trend on historical -----
        x = g[date_col].map(pd.Timestamp.toordinal).values.astype(float)
        y = g[target_col].values.astype(float)

        # Guard for very short series
        if len(g) < max(5, period):
            # minimal smoothing: moving average fallback
            lowess_fit = pd.Series(y).rolling(min(3, len(g)), min_periods=1).mean().values
        else:
            lowess_fit = lowess(y, x, frac=frac, return_sorted=False)

        g["lowess_trend"] = lowess_fit

        # ----- STL seasonality (additive), then clip to non-negative -----
        # Work on a regular monthly index to keep STL happy
        g_stl = g.set_index(date_col)
        # If freq is missing, assume monthly start
        if g_stl.index.inferred_freq is None:
            # Reindex monthly and interpolate the target for STL only
            full_idx = pd.date_range(g_stl.index.min(), g_stl.index.max(), freq="MS")
            s_for_stl = g_stl[target_col].reindex(full_idx).ffill()
        else:
            s_for_stl = g_stl[target_col]

        # Run STL on the (possibly reindexed) series
        stl = STL(s_for_stl, period=period, robust=robust_stl)
        res = stl.fit()
        seasonal_full = res.seasonal

        # Map seasonal back to original timestamps
        seasonal_hist = seasonal_full.reindex(g_stl.index).values
        # Enforce non-negative additive seasonality
        seasonal_hist = np.clip(seasonal_hist, 0.0, None)
        g["seasonality"] = seasonal_hist

        # ----- Build recency-weighted monthly seasonal index (non-negative) -----
        # Use all historical months (after clipping) grouped by month-of-year
        g["_year"] = g[date_col].dt.year
        g["_month"] = g[date_col].dt.month
        most_recent_year = g["_year"].max()

        month_to_weighted = {}
        for m in range(1, 13):
            rows_m = g[g["_month"] == m]
            if rows_m.empty:
                month_to_weighted[m] = 0.0
                continue
            vals = rows_m["seasonality"].values
            ages = (most_recent_year - rows_m["_year"].values).astype(float)  # 0 for most recent
            # exponential recency weights
            w = np.power(alpha, ages)
            # if all weights zero (edge case), fallback to equal weights
            if np.all(w == 0):
                w = np.ones_like(vals)
            month_to_weighted[m] = float(np.average(vals, weights=w))

        # ----- Extrapolate future months -----
        if extrapolate_months > 0:
            max_date = g[date_col].max()
            extra_dates = pd.date_range(start=max_date + pd.offsets.MonthBegin(1),
                                        periods=extrapolate_months, freq="MS")
            extra_x = extra_dates.map(pd.Timestamp.toordinal).values.astype(float)

            # Linear extrapolation of LOWESS trend using end slope
            if len(x) >= 2:
                slope = (lowess_fit[-1] - lowess_fit[-2]) / (x[-1] - x[-2])
            else:
                slope = 0.0
            extra_trend = lowess_fit[-1] + slope * (extra_x - x[-1])

            # Seasonality for future via recency-weighted month index
            extra_months = extra_dates.month
            extra_seasonality = np.array([month_to_weighted[m] for m in extra_months], dtype=float)

            extra_df = pd.DataFrame({
                key_col: key,
                date_col: extra_dates,
                target_col: np.nan,
                "lowess_trend": extra_trend,
                "seasonality": extra_seasonality
            })

            g = pd.concat([g[[key_col, date_col, target_col, "lowess_trend", "seasonality"]],
                           extra_df],
                          ignore_index=True)
        else:
            g = g[[key_col, date_col, target_col, "lowess_trend", "seasonality"]]

        out.append(g)

    return pd.concat(out, ignore_index=True)


def fix_negative_forecast(df, key_col, date_col, forecast_col, actual_col, cutoff_date, adjusted_col="adj_forecast"):
    """
    Replaces negative forecast values with a deterministic positive value.
    """
    df = df.copy()
    df[adjusted_col] = df[forecast_col]  # Start with original forecast

    for key in df[key_col].unique():
        key_df = df[df[key_col] == key].sort_values(date_col)

        # Split actual and forecast data
        actuals = key_df[key_df[date_col] < cutoff_date]
        forecasts = key_df[key_df[date_col] >= cutoff_date]

        if actuals.empty or forecasts.empty:
            continue

        # Get most recent actual value
        latest_actual = actuals[actual_col].dropna().iloc[-1] if not actuals[actual_col].dropna().empty else None

        # Get most recent positive forecast value
        min_positive_forecast = forecasts[forecast_col][forecasts[forecast_col] > 0].min()

        for idx in forecasts.index:
            if forecasts.at[idx, forecast_col] < 0:
                replacement_value = min_positive_forecast if min_positive_forecast > 0 else latest_actual

                if replacement_value is not None:
                    # ✅ FIX: Removed the random multiplier to ensure deterministic results.
                    df.at[idx, adjusted_col] = replacement_value
    return df


def build_monthly_upper_lower_band(
    df,
    key_col='key',
    date_col='date',
    value_col='actual_value',
    rolling_window=3,
    train_till=None):

    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
    all_results = []

    for key in df[key_col].unique():
        df_key = df[df[key_col] == key].copy().sort_values(date_col)

        # Filter training data
        train_mask = df_key[date_col] <= pd.to_datetime(train_till)
        df_train = df_key[train_mask].copy()

        # Skip if df_train is empty or its length is less than rolling_window
        if df_train.empty or len(df_train) < rolling_window:
            continue

        # Rolling min/max
        df_train['rolling_max'] = df_train[value_col].rolling(window=rolling_window, center=True).max()
        df_train['rolling_min'] = df_train[value_col].rolling(window=rolling_window, center=True).min()

        # Fill + smooth
        df_train['upper_band'] = uniform_filter1d(
            df_train['rolling_max'].ffill(),
            size=rolling_window
        )
        df_train['lower_band'] = uniform_filter1d(
            df_train['rolling_min'].ffill(),
            size=rolling_window
        )

        df_train = df_train.dropna(subset=['upper_band', 'lower_band'])
        if df_train.empty:
            continue

        # Fit trend lines
        X_train = np.arange(len(df_train)).reshape(-1, 1)
        upper_model = LinearRegression().fit(X_train, df_train['upper_band'])
        lower_model = LinearRegression().fit(X_train, df_train['lower_band'])

        # Define thresholds to avoid unrealistic predictions
        recent_actuals = df_train[value_col].dropna().values[-rolling_window:]
        min_actual_threshold = max(min(recent_actuals), 0)
        max_actual_threshold = max(recent_actuals) if len(recent_actuals) > 0 else 0

        min_lower_band_threshold = max(
            df_train['lower_band'].tail(rolling_window).min(),
            min_actual_threshold
        )
        min_upper_band_threshold = max(
            df_train['upper_band'].tail(rolling_window).min(),
            max_actual_threshold
        )

        # Forecast horizon
        full_max_date = df_key[date_col].max()
        train_last_date = df_train[date_col].max()
        months_gap = (full_max_date.to_period("M") - train_last_date.to_period("M")).n

        if months_gap <= 0:
            df_train_final = df_train[[key_col, date_col, value_col, 'upper_band', 'lower_band']]
            all_results.append(df_train_final)
            continue

        # Generate future dates
        future_dates = pd.date_range(
            start=train_last_date + pd.offsets.MonthBegin(1),
            periods=months_gap,
            freq='MS'
        )
        X_future = np.arange(len(df_train), len(df_train) + months_gap).reshape(-1, 1)

        # Predict and apply saturation
        upper_forecast_raw = upper_model.predict(X_future)
        lower_forecast_raw = lower_model.predict(X_future)

        upper_forecast = np.maximum(upper_forecast_raw, min_upper_band_threshold)
        lower_forecast = np.maximum(lower_forecast_raw, min_lower_band_threshold)

        df_future = pd.DataFrame({
            key_col: key,
            date_col: future_dates,
            value_col: np.nan,
            'upper_band': upper_forecast,
            'lower_band': lower_forecast
        })

        df_train_final = df_train[[key_col, date_col, value_col, 'upper_band', 'lower_band']]
        result_df = pd.concat([df_train_final, df_future], ignore_index=True)
        all_results.append(result_df)

    if not all_results:
        return pd.DataFrame(columns=[key_col, date_col, value_col, 'upper_band', 'lower_band'])

    return pd.concat(all_results).reset_index(drop=True)





def run_holtwinters(series, horizon, freq="MS", seasonal=None, seasonal_periods=None, trend="add"):
    """
    Fit Holt-Winters (or Holt) model depending on arguments.
    """
    if series.dropna().shape[0] < 2:
        return pd.Series(dtype=float)

    last_date = series.index[-1]
    future_dates = pd.date_range(
        last_date + pd.tseries.frequencies.to_offset(freq),
        periods=horizon, freq=freq
    )

    try:
        model = ExponentialSmoothing(
            series, 
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        fit = model.fit()
        forecast = fit.forecast(horizon)
        return pd.Series(forecast.values, index=future_dates)
    except Exception:
        return pd.Series(np.nan, index=future_dates)



def run_sarima(series, horizon, freq="MS"):
    """
    Fits the best ARIMA-family model using auto_arima.
    """
    if series.dropna().shape[0] < 15: 
        return pd.Series(dtype=float)

    last_date = series.index[-1]
    future_dates = pd.date_range(last_date + pd.tseries.frequencies.to_offset(freq),
                                 periods=horizon, freq=freq)

    seasonal_periods = {"MS": 12, "Q": 4, "W": 52}.get(freq, 12)

    try:
        model = auto_arima(series,
                           start_p=1, start_q=1, max_p=5, max_q=5,
                           m=seasonal_periods, seasonal=True, d=None, D=None,                   
                           start_P=0, start_Q=0, max_P=2, max_Q=2,         
                           stepwise=True, suppress_warnings=True, error_action='ignore')

        forecast = model.predict(n_periods=horizon)
        return pd.Series(forecast, index=future_dates)
        
    except Exception as e:
        logger.warning(f"AutoARIMA failed for series: {e}")
        return pd.Series(np.nan, index=future_dates)




def run_prophet(series, horizon, freq="MS"):
    """
    Fit Prophet model for forecasting with adaptive seasonality.
    """
    if series.dropna().shape[0] < 5:
        return pd.Series(dtype=float)
    
    last_date = series.index[-1]
    future_dates = pd.date_range(
        last_date + pd.tseries.frequencies.to_offset(freq),
        periods=horizon, 
        freq=freq
    )
    
    try:
        df_prophet = pd.DataFrame({'ds': series.index, 'y': series.values}).dropna()
        
        if len(df_prophet) < 5:
            return pd.Series(np.nan, index=future_dates)
        
        if len(df_prophet) >= 24:
            model = Prophet(
                yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode='additive', changepoint_prior_scale=0.20, 
                seasonality_prior_scale=10.0, holidays_prior_scale=10.0,
                mcmc_samples=0, interval_width=0.80)
        else:
            model = Prophet(
                yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode='additive', changepoint_prior_scale=0.1,  
                seasonality_prior_scale=10.0, mcmc_samples=0, interval_width=0.80)
            
            data_points = len(df_prophet)
            fourier_order = min(max(1, data_points // 4), 6, 3)
            
            model.add_seasonality(
                name='monthly_custom', period=12, fourier_order=fourier_order, prior_scale=10.0)
            
        model.fit(df_prophet)
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        forecasted_values = np.maximum(forecast['yhat'].values, 0.0)
        
        return pd.Series(forecasted_values, index=future_dates)
        
    except Exception as e:
        logger.warning(f"Prophet failed for series: {e}")
        return pd.Series(np.nan, index=future_dates)


def run_ly_trend(series, horizon, freq="MS", **kwargs):
    """
    The final, fully integrated, and corrected version of the LY_trend model.
    """
    key_col = kwargs['key_col']
    date_col = kwargs['date_col']
    target_col = kwargs['target_col']
    key_name = kwargs['key_name']

    df_input = series.to_frame(name=target_col).reset_index()
    df_input.rename(columns={'index': date_col}, inplace=True)
    df_input[key_col] = key_name

    if len(df_input) < 13:
        future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq=freq)
        return pd.Series(np.nan, index=future_dates)
    
    try:
        cutoff_date = df_input[date_col].max()
        
        df_decomposed = lowess_trend_seasonality_extrapolate(
            df=df_input, key_col=key_col, date_col=date_col, target_col=target_col,
            extrapolate_months=12, frac=0.6, period=12, alpha=0.5)
        
        df_decomposed = fix_negative_forecast(
            df_decomposed, key_col=key_col, date_col=date_col,
            forecast_col='lowess_trend', actual_col=target_col,
            cutoff_date=cutoff_date, adjusted_col='lowess_trend_adj')
        
        df_decomposed = df_decomposed.sort_values([key_col, date_col])
        df_decomposed['LY'] = df_decomposed.groupby(key_col)[target_col].shift(12)
        df_decomposed['year'] = df_decomposed[date_col].dt.year

        yearly_df_input = df_decomposed.groupby([key_col, 'year'])[target_col].sum().reset_index()
        yearly_df_input.rename(columns={target_col: 'Yearly_vol'}, inplace=True)
        yearly_df_input['year'] = yearly_df_input['year'] + 1
        
        df_decomposed = df_decomposed.merge(yearly_df_input, on=[key_col, 'year'], how='left')
        
        df_decomposed['year_distr'] = df_decomposed['LY'] / df_decomposed['Yearly_vol']
        df_decomposed['year_distr'] = df_decomposed.groupby(key_col)['year_distr'].ffill()

        annual_trend = df_decomposed.groupby([key_col, 'year'])['lowess_trend_adj'].sum().reset_index()
        annual_trend.rename(columns={'lowess_trend_adj': 'annual_trend'}, inplace=True)
        
        df_decomposed = df_decomposed.merge(annual_trend, on=[key_col, 'year'], how='left')
        
        df_decomposed['annual_trend'] = df_decomposed.groupby(key_col)['annual_trend'].ffill()

        df_decomposed['Trend_LY'] = (df_decomposed['annual_trend'] * df_decomposed['year_distr'])

        future_dates = pd.date_range(cutoff_date + pd.DateOffset(months=1), periods=horizon, freq=freq)
        
        forecast_df = df_decomposed[df_decomposed[date_col].isin(future_dates)]
        
        if forecast_df.empty:
            return pd.Series(np.nan, index=future_dates)
        
        forecast_series = forecast_df.set_index(date_col)['Trend_LY']
        
        return forecast_series.reindex(future_dates)

    except Exception as e:
        logger.warning(f"LY_trend model failed for series {key_name}: {e}")
        future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq=freq)
        return pd.Series(np.nan, index=future_dates)

    

def run_l3m(train_series, forecast_dates):
    preds = {}
    for d in forecast_dates:
        l3m_start = d - pd.DateOffset(months=3)
        l3m_data = train_series[(train_series.index >= l3m_start) & (train_series.index < d)].dropna()

        if len(l3m_data) > 0:
            preds[d] = l3m_data.mean()
        else:
            preds[d] = train_series.dropna().iloc[-1] if len(train_series.dropna()) > 0 else np.nan

    return pd.Series(preds, index=pd.to_datetime(forecast_dates))


def run_l6m(train_series, forecast_dates):
    preds = {}
    for d in forecast_dates:
        l6m_start = d - pd.DateOffset(months=6)
        l6m_data = train_series[(train_series.index >= l6m_start) & (train_series.index < d)].dropna()

        if len(l6m_data) > 0:
            preds[d] = l6m_data.mean()
        else:
            preds[d] = train_series.dropna().iloc[-1] if len(train_series.dropna()) > 0 else np.nan

    return pd.Series(preds, index=pd.to_datetime(forecast_dates))


def run_naive(series, horizon, freq="MS", **kwargs):
    """
    Naive forecast that uses the last known, non-null value from the training series.
    """
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=horizon, 
        freq=freq
    )
    
    train_series_no_nulls = series.dropna()
    
    if train_series_no_nulls.empty:
        last_value = np.nan
    else:
        last_value = train_series_no_nulls.iloc[-1]
        
    return pd.Series(last_value, index=future_dates)


def get_prophet_trend_components(series, train_only_cutoff=None):
    """
    Fits a Prophet model and returns trend components and the fitted model object.
    """
    series = series.copy().sort_index()
    train_series = series[series.index <= train_only_cutoff] if train_only_cutoff else series
    train_series_clean = train_series.dropna()

    if len(train_series_clean) < 5:
        return pd.DataFrame({'trend': np.nan}, index=series.index), None

    try:
        df_prophet = train_series_clean.reset_index()
        df_prophet.columns = ['ds', 'y']

        model = Prophet(
            yearly_seasonality=len(df_prophet) >= 24,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.25 if len(df_prophet) >= 24 else 0.1
        )
        if len(df_prophet) < 24:
             model.add_seasonality(name='monthly_custom', period=12, fourier_order=3)
        
        model.fit(df_prophet)

        full_df_prophet = pd.DataFrame({'ds': series.index})
        full_forecast = model.predict(full_df_prophet)
        full_trend = pd.Series(full_forecast['trend'].values, index=series.index)
        
        return pd.DataFrame({'trend': full_trend}), model

    except Exception as e:
        logger.warning(f"Prophet trend extraction failed: {e}")
        return pd.DataFrame({'trend': np.nan}, index=series.index), None


def add_trend_seasonality(series, period=12, frac=0.3, alpha=0.85, robust_stl=True, use_prophet_trend=True, train_only_cutoff=None):
    """
    Computes trend and seasonality, now correctly attaching the fitted Prophet model.
    """
    series = series.copy().sort_index()
    df = pd.DataFrame({"target": series})
    prophet_model = None

    if use_prophet_trend:
        trend_df, prophet_model = get_prophet_trend_components(series, train_only_cutoff=train_only_cutoff)
        df["trend"] = trend_df["trend"]

    if 'trend' not in df.columns or df["trend"].isna().all():
        from statsmodels.nonparametric.smoothers_lowess import lowess
        train_series = series[series.index <= train_only_cutoff] if train_only_cutoff else series
        if len(train_series.dropna()) >= 5:
             x = train_series.dropna().index.map(pd.Timestamp.toordinal)
             y = train_series.dropna().values
             trend_values = lowess(y, x, frac=frac, return_sorted=False)
             df.loc[train_series.dropna().index, 'trend'] = trend_values
        df['trend'] = df['trend'].interpolate(method='linear').ffill().bfill()


    train_series_for_stl = series[series.index <= train_only_cutoff] if train_only_cutoff else series
    if len(train_series_for_stl.dropna()) >= 2 * period:
        stl = STL(train_series_for_stl.dropna().asfreq('MS'), period=period, robust=robust_stl)
        res = stl.fit()
        seasonal_hist = res.seasonal
        
        train_df = pd.DataFrame({"seasonality": seasonal_hist})
        train_df["_month"] = train_df.index.month
        month_to_weighted = train_df.groupby("_month")["seasonality"].mean().to_dict()
    else: 
        month_to_weighted = {m: 0.0 for m in range(1, 13)}

    df["seasonality"] = df.index.month.map(month_to_weighted).fillna(0.0)
    df.attrs["month_to_weighted"] = month_to_weighted
    df.attrs["prophet_model"] = prophet_model
    return df



def prepare_features(series, freq="MS", extra_features=None, train_only_cutoff=None, use_prophet_trend=True, monthly_ratios_df=None, key_name=None):
    """
    Create lag, rolling, calendar, trend, seasonality features from Series.
    """
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    df = add_trend_seasonality(
        series, period=12, frac=0.3, alpha=0.85, robust_stl=True, 
        use_prophet_trend=use_prophet_trend, train_only_cutoff=train_only_cutoff
    )
    
    month_to_weighted = df.attrs["month_to_weighted"]

    for lag in [1, 2, 3]:
        df[f"lag{lag}"] = df["target"].shift(lag)

    df["rolling_mean_3m"] = df["target"].shift(1).rolling(3).mean()
    df["rolling_mean_6m"] = df["target"].shift(1).rolling(6).mean()
    df["rolling_std_3m"] = df["target"].shift(1).rolling(3).std()

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["month_end_flag"] = df["month"].isin([3, 6, 9, 12]).astype(int)

    if monthly_ratios_df is not None and key_name is not None:
        key_ratios = monthly_ratios_df[monthly_ratios_df[monthly_ratios_df.columns[0]] == key_name].copy()
        if not key_ratios.empty:
            month_to_ratio = key_ratios.set_index('month')['avg_monthly_ratio'].to_dict()
            df["monthly_ratio"] = df["month"].map(month_to_ratio).fillna(1/12)
        else:
            df["monthly_ratio"] = 1/12
    
    if extra_features is not None:
        ef = extra_features.copy()
        ef["date"] = pd.to_datetime(ef["date"]).dt.to_period("M").dt.to_timestamp("MS")
        ef = ef.groupby("date", as_index=False).mean(numeric_only=True)
        df = df.reset_index().rename(columns={"index": "date"})
        df = df.merge(ef, on="date", how="left").set_index("date")

    df = df.dropna()
    df.attrs["month_to_weighted"] = month_to_weighted
    return df


def prepare_features_direct(series, horizon=3, train_only_cutoff=None, monthly_ratios_df=None, key_name=None):
    """
    FIXED: Signature now accepts explicit arguments instead of **kwargs to prevent TypeErrors.
    """
    df = add_trend_seasonality(series, train_only_cutoff=train_only_cutoff)
    
    for lag in [1, 2, 3]:
        df[f"lag{lag}"] = df["target"].shift(lag)

    df["rolling_mean_3m"] = df["target"].shift(1).rolling(3).mean()
    df["rolling_mean_6m"] = df["target"].shift(1).rolling(6).mean()
    df["rolling_std_3m"] = df["target"].shift(1).rolling(3).std()

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    
    if monthly_ratios_df is not None and key_name is not None:
        key_ratios_mask = monthly_ratios_df[monthly_ratios_df.columns[0]] == key_name
        key_ratios = monthly_ratios_df[key_ratios_mask]
        if not key_ratios.empty:
            month_to_ratio = key_ratios.set_index('month')['avg_monthly_ratio'].to_dict()
            df["monthly_ratio"] = df["month"].map(month_to_ratio).fillna(1/12)
        else:
            df["monthly_ratio"] = 1/12

    for h in range(1, horizon + 1):
        df[f"target_h{h}"] = df["target"].shift(-h)

    return df


def _direct_forecast_loop(df, models, feature_cols, future_dates, horizon, monthly_ratios_df=None):
    prophet_model = df.attrs.get("prophet_model")
    if not prophet_model:
        raise ValueError("Prophet model not found for future trend projection.")

    future_df_prophet = pd.DataFrame({'ds': future_dates})
    future_components = prophet_model.predict(future_df_prophet)

    X_pred_base = df[feature_cols].iloc[[-1]].copy().fillna(0)
    preds = []

    for i, f_date in enumerate(future_dates):
        h = i + 1
        X_pred_h = X_pred_base.copy()
        
        X_pred_h['year'] = f_date.year
        X_pred_h['month'] = f_date.month
        X_pred_h['quarter'] = f_date.quarter
        X_pred_h['trend'] = future_components.iloc[i]['trend']
        X_pred_h['seasonality'] = df.attrs["month_to_weighted"].get(f_date.month, 0.0)
        
        if 'monthly_ratio' in X_pred_h.columns and monthly_ratios_df is not None:
             X_pred_h['monthly_ratio'] = monthly_ratios_df.set_index('month')['avg_monthly_ratio'].get(f_date.month, 1/12)

        # Ensure there are no NaNs before predicting, especially for rolling features
        X_pred_h.fillna(0, inplace=True)

        y_pred = models[h].predict(X_pred_h[feature_cols])[0]
        preds.append(y_pred)

    return pd.Series(preds, index=future_dates)


def run_rf_direct(series, horizon=3, freq="MS", train_cutoff=None, **kwargs):
    
    monthly_ratios_df = get_monthly_ratios_for_cutoff(kwargs['full_df'], kwargs['key_name'], train_cutoff, kwargs['key_col'], kwargs['target_col'], kwargs['date_col'])
    
    df = prepare_features_direct(
        series, horizon=horizon, train_only_cutoff=train_cutoff,
        monthly_ratios_df=monthly_ratios_df, key_name=kwargs.get('key_name'))
    
    target_cols = [f"target_h{h}" for h in range(1, horizon + 1)]
    feature_cols = [col for col in df.columns if col not in ['target'] + target_cols]
    df_train = df.dropna(subset=target_cols)
    future_dates = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
    
    if df_train.empty: return pd.Series(np.nan, index=future_dates), None

    X_train = df_train[feature_cols].fillna(0)
    models = {}
    importances_per_horizon = []
    for h in range(1, horizon + 1):
        model_params = {k: v for k, v in kwargs.items() if k in RandomForestRegressor().get_params()}
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **model_params)
        model.fit(X_train, df_train[f"target_h{h}"])
        models[h] = model
        importances_per_horizon.append(dict(zip(X_train.columns, model.feature_importances_)))

    avg_importances = pd.DataFrame(importances_per_horizon).mean().to_dict()
    predictions = _direct_forecast_loop(df, models, feature_cols, future_dates, horizon, monthly_ratios_df)
    return predictions, avg_importances


def run_xgb_direct(series, horizon=3, freq="MS", train_cutoff=None, **kwargs):
    
    monthly_ratios_df = get_monthly_ratios_for_cutoff(kwargs['full_df'], kwargs['key_name'], train_cutoff, kwargs['key_col'], kwargs['target_col'], kwargs['date_col'])

    df = prepare_features_direct(
        series, horizon=horizon, train_only_cutoff=train_cutoff,
        monthly_ratios_df=monthly_ratios_df, key_name=kwargs.get('key_name'))
    
    target_cols = [f"target_h{h}" for h in range(1, horizon + 1)]
    feature_cols = [col for col in df.columns if col not in ['target'] + target_cols]
    df_train = df.dropna(subset=target_cols)
    future_dates = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)

    if df_train.empty: return pd.Series(np.nan, index=future_dates), None

    X_train = df_train[feature_cols].fillna(0)
    models = {}
    importances_per_horizon = []
    for h in range(1, horizon + 1):
        model_params = {k: v for k, v in kwargs.items() if k in XGBRegressor().get_params()}
        model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **model_params)
        model.fit(X_train, df_train[f"target_h{h}"])
        models[h] = model
        importances_per_horizon.append(dict(zip(X_train.columns, model.feature_importances_)))

    avg_importances = pd.DataFrame(importances_per_horizon).mean().to_dict()
    predictions = _direct_forecast_loop(df, models, feature_cols, future_dates, horizon, monthly_ratios_df)
    return predictions, avg_importances

def run_lgbm_direct(series, horizon=3, freq="MS", train_cutoff=None, **kwargs):
    """
    Direct forecasting model using LightGBM.
    """
    monthly_ratios_df = get_monthly_ratios_for_cutoff(kwargs['full_df'], kwargs['key_name'], train_cutoff, kwargs['key_col'], kwargs['target_col'], kwargs['date_col'])

    df = prepare_features_direct(
        series, horizon=horizon, train_only_cutoff=train_cutoff,
        monthly_ratios_df=monthly_ratios_df, key_name=kwargs.get('key_name'))
    
    target_cols = [f"target_h{h}" for h in range(1, horizon + 1)]
    feature_cols = [col for col in df.columns if col not in ['target'] + target_cols]
    df_train = df.dropna(subset=target_cols)
    future_dates = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)

    if df_train.empty: return pd.Series(np.nan, index=future_dates), None

    X_train = df_train[feature_cols].fillna(0)
    models = {}
    importances_per_horizon = []
    for h in range(1, horizon + 1):
        model_params = {k: v for k, v in kwargs.items() if k in LGBMRegressor().get_params()}
        default_lgbm_params = {'objective': 'regression_l1', 'random_state': 42, 'n_jobs': -1, 'num_leaves': 10, 'min_child_samples': 5}
        final_params = {**default_lgbm_params, **model_params}

        model = LGBMRegressor(**final_params, verbose=-1)
        model.fit(X_train, df_train[f"target_h{h}"])
        models[h] = model
        importances_per_horizon.append(dict(zip(X_train.columns, model.feature_importances_)))
    
    avg_importances = pd.DataFrame(importances_per_horizon).mean().to_dict()
    predictions = _direct_forecast_loop(df, models, feature_cols, future_dates, horizon, monthly_ratios_df)

    return predictions, avg_importances


def tune_rf_hyperparameters_optuna_direct(train_series, full_df, key_name, key_col, target_col, date_col, horizon=3):
    logger.info(f"Tuning Direct RF with Optuna for key: {key_name}...")
    try:
        monthly_ratios_df = get_monthly_ratios_for_cutoff(full_df, key_name, train_series.index.max(), key_col, target_col, date_col or 'date')
        df_multi_target = prepare_features_direct(
            train_series, horizon=horizon, train_only_cutoff=train_series.index.max(),
            monthly_ratios_df=monthly_ratios_df, key_name=key_name)
        
        target_cols = [f"target_h{h}" for h in range(1, horizon + 1)]
        feature_cols = [col for col in df_multi_target.columns if col not in ['target'] + target_cols]
        df_train = df_multi_target.dropna(subset=target_cols)
        X_train, y_train = df_train[feature_cols], df_train[target_cols]
        X_train = X_train.fillna(0)

        if X_train.empty or len(X_train) < 15:
            logger.warning("Skipping tuning: Not enough data for cross-validation.")
            return {}

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.8]),
                'random_state': 42, 'bootstrap': True
            }
            tscv, horizon_errors = TimeSeriesSplit(n_splits=3), []
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                fold_errors = []
                for h in range(1, horizon + 1):
                    model = RandomForestRegressor(**params, n_jobs=4)
                    model.fit(X_fold_train, y_fold_train[f'target_h{h}'])
                    preds = model.predict(X_fold_val)
                    fold_errors.append(mean_absolute_error(y_fold_val[f'target_h{h}'], preds))
                horizon_errors.append(np.mean(fold_errors))
            return np.mean(horizon_errors)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25, n_jobs=6)
        logger.info(f"Best Direct RF params for key {key_name}: {study.best_params}")
        return study.best_params
    except Exception as e:
        logger.error(f"Direct RF Optuna tuning failed for key {key_name}: {e}. Using default parameters.")
        return {}

def tune_xgb_hyperparameters_optuna_direct(train_series, full_df, key_name, key_col, target_col, date_col, horizon=3):
    """
    Finds the best hyperparameters for a direct XGBoost model using Optuna.
    """
    logger.info(f"Tuning Direct XGB with Optuna for key: {key_name}...")
    try:
        monthly_ratios_df = get_monthly_ratios_for_cutoff(
            full_df, key_name, train_series.index.max(), key_col, target_col, date_col or 'date')
        df_multi_target = prepare_features_direct(
            train_series, horizon=horizon, train_only_cutoff=train_series.index.max(),
            monthly_ratios_df=monthly_ratios_df, key_name=key_name)

        target_cols = [f"target_h{h}" for h in range(1, horizon + 1)]
        feature_cols = [col for col in df_multi_target.columns if col not in ['target'] + target_cols]
        df_train = df_multi_target.dropna(subset=target_cols)
        
        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_cols]

        if X_train.empty or len(X_train) < 15:
            logger.warning(f"Skipping XGB tuning for key {key_name}: Not enough data.")
            return {}

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'objective': 'reg:squarederror', 'random_state': 42
            }
            tscv = TimeSeriesSplit(n_splits=3)
            horizon_errors = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                fold_errors = []
                for h in range(1, horizon + 1):
                    model = XGBRegressor(**params, n_jobs=4)
                    model.fit(X_fold_train, y_fold_train[f'target_h{h}'])
                    preds = model.predict(X_fold_val)
                    fold_errors.append(mean_absolute_error(y_fold_val[f'target_h{h}'], preds))
                horizon_errors.append(np.mean(fold_errors))
            return np.mean(horizon_errors)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, n_jobs=6)
        
        logger.info(f"Best Direct XGB params for key {key_name}: {study.best_params}")
        return study.best_params

    except Exception as e:
        logger.error(f"Direct XGB Optuna tuning failed for key {key_name}: {e}. Using default parameters.")
        return {}


def tune_lgbm_hyperparameters_optuna_direct(train_series, full_df, key_name, key_col, target_col, date_col, horizon=3):
    """
    Finds the best hyperparameters for a direct LightGBM model using Optuna.
    """
    logger.info(f"Tuning Direct LGBM with Optuna for key: {key_name}...")
    try:
        monthly_ratios_df = get_monthly_ratios_for_cutoff(
            full_df, key_name, train_series.index.max(), key_col, target_col, date_col or 'date')
        df_multi_target = prepare_features_direct(
            train_series, horizon=horizon, train_only_cutoff=train_series.index.max(),
            monthly_ratios_df=monthly_ratios_df, key_name=key_name)

        target_cols = [f"target_h{h}" for h in range(1, horizon + 1)]
        feature_cols = [col for col in df_multi_target.columns if col not in ['target'] + target_cols]
        df_train = df_multi_target.dropna(subset=target_cols)
        
        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_cols]

        if X_train.empty or len(X_train) < 15:
            logger.warning(f"Skipping LGBM tuning for key {key_name}: Not enough data.")
            return {}

        def objective(trial):
            params = {
                'objective': 'regression_l1', 'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 5, 20),
                'max_depth': -1,
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            }            
            tscv = TimeSeriesSplit(n_splits=3)
            horizon_errors = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                fold_errors = []
                for h in range(1, horizon + 1):
                    model = LGBMRegressor(**params, n_jobs=4, verbose=-1)
                    model.fit(X_fold_train, y_fold_train[f'target_h{h}'])
                    preds = model.predict(X_fold_val)
                    fold_errors.append(mean_absolute_error(y_fold_val[f'target_h{h}'], preds))
                horizon_errors.append(np.mean(fold_errors))
            return np.mean(horizon_errors)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, n_jobs=6)
        
        logger.info(f"Best Direct LGBM params for key {key_name}: {study.best_params}")
        return study.best_params

    except Exception as e:
        logger.error(f"Direct LGBM Optuna tuning failed for key {key_name}: {e}. Using default parameters.")
        return {}



def clean_series_no_leakage(series, cutoff_date):
    """
    Cleans series up to cutoff_date by ensuring a complete monthly index
    and filling missing values AFTER the first valid data point.
    """
    first_valid_index = series.first_valid_index()
    if first_valid_index is None:
        return series[series.index <= cutoff_date]

    full_index = pd.date_range(start=first_valid_index, end=cutoff_date, freq='MS')
    series_reindexed = series.reindex(full_index)
    series_filled = series_reindexed.fillna(0)
    series_after_cutoff = series[series.index > cutoff_date]
    final_series = pd.concat([series_filled, series_after_cutoff])
    return final_series



def add_hist_range(df, key_col='key', date_col='date'):
    """
    Calculates the history length for each key and adds the 'hist_range' column.
    """
    df_processed = df.copy()
    history_counts = df_processed.groupby(key_col)[date_col].count()

    def assign_range(num_months):
        if num_months >= 36: return "36+"
        elif 25 <= num_months <= 35: return "25–36"
        elif 13 <= num_months <= 24: return "13–24"
        elif 7 <= num_months <= 12: return "7–12"
        else: return "<6"

    range_map = history_counts.apply(assign_range)
    df_processed['hist_range'] = df_processed[key_col].map(range_map)
    return df_processed


# Model Execution Switcher

MODEL_FUNCS = {
    "RF": run_rf_direct, "XGB": run_xgb_direct, "LGBM": run_lgbm_direct,
    "RF_default": run_rf_direct, "XGB_default": run_xgb_direct, "LGBM_default": run_lgbm_direct,
    "HoltWinters": run_holtwinters, "SARIMA": run_sarima, "Prophet": run_prophet,
    "TBF": run_TBF, "L3M": run_l3m, "L6M": run_l6m,
    "LY_trend": run_ly_trend, "Naive": run_naive
}



def map_hist_range_to_bucket(hist_range):
    """Correctly map hist_range to hist_bucket"""
    if hist_range in ["36+", "25–36"]: return "24+"
    elif hist_range == "13–24": return "13–24"
    elif hist_range == "7–12": return "7–12"
    else: return "<6"
    

def forecast_pipeline_debug(
    df: pd.DataFrame,
    parameters: Dict,
    validation_cutoff: str,
    test_cutoff: str,
    forecast_cutoff: str,
    forecasting_horizon: int = 3,
    debug_keys: Optional[List[str]] = None,
    max_debug_logs_per_key: int = 1000
) -> pd.DataFrame:
    """
    Corrected version of the forecast pipeline with integrated logging.
    """
    date_col, target_col, key_col = parameters["date_col"], parameters["target_col"], parameters["key_col"]
    seasonal_col, hist_range_col = parameters["seasonal_col"], parameters["hist_range_col"]
    hierarchy_cols: List[str] = parameters.get("hierarchy_cols", [])

    validation_cutoff = pd.to_datetime(validation_cutoff)
    test_cutoff = pd.to_datetime(test_cutoff)
    forecast_cutoff = pd.to_datetime(forecast_cutoff)
    df = df.copy()

    all_results, all_feature_importances = [], []
    all_model_names = ["SARIMA", "HoltWinters", "Prophet", "L3M", "L6M", "RF", "XGB", "LY_trend", "TBF", "Naive", "LGBM", "RF_default", "XGB_default", "LGBM_default"]
    
    # Define once, early, so it always exists regardless of later control flow
    feature_cols_to_add = [
        "trend", "seasonality", "lag1", "lag2", "lag3",
        "rolling_mean_3m", "rolling_mean_6m", "rolling_std_3m",
        "year", "month", "quarter", "monthly_ratio"
    ]

    keys = list(df[key_col].unique())
    if debug_keys is not None:
        keys = [k for k in keys if k in set(debug_keys)]
        if not keys:
            logger.warning("Debug_keys provided but none matched dataset keys -> returning empty df")
            return pd.DataFrame()

    for key in keys:
        # <--- ADDED LOGGING: Main try/except block for key-level error handling --->
        try:
            logger.info(f"--- Processing Key: {key} ---")
            group = df[df[key_col] == key].copy()

            if seasonal_col not in group.columns or hist_range_col not in group.columns:
                logger.error(f"Missing seasonal/hist_range for key {key}; skipping.")
                continue

            # Normalize seasonal flag to 'Y'/'N' regardless of input values
            seasonal_raw = group[seasonal_col].iloc[0]
            seasonal = 'Y' if str(seasonal_raw).upper().startswith('Y') else 'N'
            hist_range = group[hist_range_col].iloc[0]
            hier_values = {c: group[c].iloc[0] if c in group.columns else None for c in hierarchy_cols}

            hist_bucket = map_hist_range_to_bucket(hist_range)
            models_to_run = globals().get("MODEL_RULES", {}).get((seasonal, hist_bucket), [])
            
            # <--- ADDED LOGGING: Log metadata and models to run --->
            logger.info(f"  Metadata -> seasonal={seasonal}, hist_range={hist_range}, hist_bucket={hist_bucket}")
            logger.info(f"  Models to run: {models_to_run}")

            if not models_to_run:
                logger.info("  SKIP: no models configured for these meta values.")
                continue

            group = group.sort_values(date_col)
            group[date_col] = pd.to_datetime(group[date_col])
            group = group.set_index(date_col)
            group[target_col] = group[target_col].astype(float)

            validation_dates = pd.date_range(validation_cutoff + pd.offsets.MonthBegin(1), test_cutoff, freq="MS")
            test_dates = pd.date_range(test_cutoff + pd.offsets.MonthBegin(1), forecast_cutoff, freq="MS")
            future_dates = pd.date_range(forecast_cutoff + pd.offsets.MonthBegin(1), periods=forecasting_horizon, freq="MS")

            series_to_clean = group[target_col]
            cleaned_to_val = series_to_clean.copy()
            cleaned_to_val.loc[cleaned_to_val.index <= validation_cutoff] = cleaned_to_val.loc[cleaned_to_val.index <= validation_cutoff].fillna(0)
            cleaned_to_test = series_to_clean.copy()
            cleaned_to_test.loc[cleaned_to_test.index <= test_cutoff] = cleaned_to_test.loc[cleaned_to_test.index <= test_cutoff].fillna(0)
            cleaned_to_forecast = series_to_clean.copy()
            cleaned_to_forecast.loc[cleaned_to_forecast.index <= forecast_cutoff] = cleaned_to_forecast.loc[cleaned_to_forecast.index <= forecast_cutoff].fillna(0)

            train_data = cleaned_to_val[cleaned_to_val.index <= validation_cutoff].dropna()
            validation_data = cleaned_to_val[cleaned_to_val.index.isin(validation_dates)].dropna()
            test_data = cleaned_to_test[cleaned_to_test.index.isin(test_dates)].dropna()

            logger.info(f"  Data splits -> train:{len(train_data)}, validation:{len(validation_data)}, test:{len(test_data)}")

            if len(train_data) < 3 or validation_data.empty:
                # Always produce a fallback forecast, even if test data is empty
                logger.warning(f"  Insufficient training/validation data for key {key}. Using Naive fallback.")
                
                full_history_series = group[target_col]
                min_date = full_history_series.index.min() if not full_history_series.empty else forecast_cutoff
                full_dates = pd.date_range(min_date, future_dates.max(), freq="MS")
                
                base_df = pd.DataFrame({"date": full_dates, key_col: key, seasonal_col: seasonal, hist_range_col: hist_range, **hier_values})
                base_df = base_df.merge(full_history_series.to_frame(name="actual_value"), left_on="date", right_index=True, how="left")

                # Forecast future using all available historical data up to forecast_cutoff
                series_for_forecast = full_history_series[full_history_series.index <= forecast_cutoff]
                if not series_for_forecast.dropna().empty:
                    preds_future = run_naive(series_for_forecast, forecasting_horizon)
                    base_df.loc[base_df["date"].isin(future_dates), "fcst_best_raw_model_unadjusted"] = preds_future.values

                base_df['best_model_raw'] = 'Naive'
                base_df['best_model_bias_adj'] = 'Naive'
                base_df['fcst_best_raw_model_adjusted'] = base_df['fcst_best_raw_model_unadjusted']
                base_df['fcst_best_adj_model_adjusted'] = base_df['fcst_best_raw_model_unadjusted']

                val_mask = base_df["date"].isin(validation_dates)
                test_mask = base_df["date"].isin(test_dates)
                future_mask = base_df["date"].isin(future_dates)
                base_df['period'] = np.select([val_mask, test_mask, future_mask], ['validation', 'testing', 'forecasting'], default='train')
                
                all_results.append(base_df)
                logger.info(f"--- Finished processing Key: {key} ---")
                continue

            tuned_params = {}
            if 'RF' in models_to_run:
                tuned_params['RF'] = tune_rf_hyperparameters_optuna_direct(
                    train_series=train_data, full_df=df, key_name=key, key_col=key_col,
                    target_col=target_col, date_col=date_col, horizon=forecasting_horizon)
            if 'XGB' in models_to_run:
                tuned_params['XGB'] = tune_xgb_hyperparameters_optuna_direct(
                    train_series=train_data, full_df=df, key_name=key, key_col=key_col,
                    target_col=target_col, date_col=date_col, horizon=forecasting_horizon)
            if 'LGBM' in models_to_run:
                tuned_params['LGBM'] = tune_lgbm_hyperparameters_optuna_direct(
                    train_series=train_data, full_df=df, key_name=key, key_col=key_col,
                    target_col=target_col, date_col=date_col, horizon=forecasting_horizon)

            full_dates = pd.date_range(train_data.index.min(), future_dates.max(), freq="MS")
            base_df = pd.DataFrame({"date": full_dates, key_col: key, seasonal_col: seasonal, hist_range_col: hist_range, **hier_values})
            actual_values_series = cleaned_to_forecast.to_frame(name="actual_value") if isinstance(cleaned_to_forecast, pd.Series) else cleaned_to_forecast.rename(columns={target_col: "actual_value"})
            base_df = base_df.merge(actual_values_series, left_on="date", right_index=True, how="left")
            
            for m in all_model_names:
                base_df[f"{m}_pred"] = np.nan
                base_df[f"{m}_adj_pred"] = np.nan

            val_preds_dict = {}
            for model_name in models_to_run:
                try:
                    model_func = MODEL_FUNCS.get(model_name)
                    if model_name in ["L3M", "L6M"]:
                        preds_val = model_func(train_data, validation_dates)
                    elif "direct" in model_func.__name__:
                        preds_val, _ = model_func(train_data, len(validation_dates), train_cutoff=validation_cutoff, full_df=df, key_name=key,key_col=key_col,target_col=target_col,date_col=date_col, **tuned_params.get(model_name.replace('_default',''), {}))
                    elif model_name in ["TBF", "LY_trend"]:
                        preds_val = model_func(train_data, len(validation_dates), train_cutoff=validation_cutoff, full_df=df, key_name=key,key_col=key_col,target_col=target_col,date_col=date_col,ma_order=9, rolling_window=6)
                    else:
                        preds_val = model_func(train_data, len(validation_dates))
                    
                    preds_series = pd.Series(np.ravel(preds_val), index=validation_dates) if preds_val is not None else pd.Series(dtype=float, index=validation_dates)
                    base_df.loc[base_df["date"].isin(validation_dates), f"{model_name}_pred"] = preds_series.values
                    val_preds_dict[model_name] = preds_series
                except Exception as e:
                    # <--- ADDED LOGGING: Log exceptions during model runs --->
                    logger.warning(f"  EXCEPTION running {model_name} on validation for key {key}: {e!r}")

            train_data_test = cleaned_to_test[cleaned_to_test.index < test_dates.min()].dropna()
            for model_name in models_to_run:
                try:
                    model_func = MODEL_FUNCS.get(model_name)
                    if model_name in ["L3M", "L6M"]:
                        preds_test = model_func(train_data_test, test_dates)
                    elif "direct" in model_func.__name__:
                        preds_test, _ = model_func(train_data_test, len(test_dates), train_cutoff=test_cutoff, full_df=df, key_name=key,key_col=key_col,target_col=target_col,date_col=date_col, **tuned_params.get(model_name.replace('_default',''), {}))
                    elif model_name in ["TBF", "LY_trend"]:
                        preds_test = model_func(train_data_test, len(test_dates), train_cutoff=test_cutoff, full_df=df, key_name=key,key_col=key_col,target_col=target_col,date_col=date_col,ma_order=9, rolling_window=6)
                    else:
                        preds_test = model_func(train_data_test, len(test_dates))
                        
                    preds_test_series = pd.Series(np.ravel(preds_test), index=test_dates) if preds_test is not None else pd.Series(dtype=float, index=test_dates)
                    base_df.loc[base_df["date"].isin(test_dates), f"{model_name}_pred"] = preds_test_series.values
                    adj_preds_test = calculate_bias_adjustment(base_df.loc[base_df["date"].isin(validation_dates), "actual_value"].dropna(), val_preds_dict.get(model_name, pd.Series(dtype=float)), preds_test_series)
                    base_df.loc[base_df["date"].isin(test_dates), f"{model_name}_adj_pred"] = adj_preds_test.values
                except Exception as e:
                    logger.warning(f"  EXCEPTION running {model_name} in testing for key {key}: {e!r}")

            model_errors = []
            for model_name in models_to_run:
                val_actuals = base_df.loc[base_df['date'].isin(validation_dates), 'actual_value'].fillna(0)
                val_preds = base_df.loc[base_df['date'].isin(validation_dates), f'{model_name}_pred'].fillna(0)
                val_error = mean_absolute_error(val_actuals, val_preds) if not val_actuals.empty else np.inf
                
                test_actuals = base_df.loc[base_df['date'].isin(test_dates), 'actual_value'].fillna(0)
                test_raw_preds = base_df.loc[base_df['date'].isin(test_dates), f'{model_name}_pred'].fillna(0)
                test_error = mean_absolute_error(test_actuals, test_raw_preds) if not test_actuals.empty else np.inf
                
                test_adj_preds = base_df.loc[base_df['date'].isin(test_dates), f'{model_name}_adj_pred'].fillna(0)
                adj_test_error = mean_absolute_error(test_actuals, test_adj_preds) if not test_actuals.empty else np.inf

                model_errors.append({"model": model_name, "val_error": val_error, "test_error": test_error, "adj_test_error": adj_test_error, "total_error_raw": val_error + test_error, "total_error_adj": val_error + adj_test_error})

            valid_forecast_models = [m for m in model_errors if np.isfinite(m.get("total_error_raw", np.inf))]
            if not valid_forecast_models:
                logger.warning(f"  SKIP: No valid forecast models found for key {key}.")
                continue

            best_model_raw = min(valid_forecast_models, key=lambda x: x["total_error_raw"])["model"]
            best_model_bias_adj = min(valid_forecast_models, key=lambda x: x["total_error_adj"])["model"]
            
            # <--- ADDED LOGGING: Log best model selection --->
            logger.info(f"  Best Raw Model (Method A): {best_model_raw}")
            logger.info(f"  Best Bias-Adjusted Model (Method B): {best_model_bias_adj}")

            train_data_forecast = cleaned_to_forecast.dropna()
            final_forecasts = {}
            for model_name in models_to_run:
                try:
                    model_func = MODEL_FUNCS.get(model_name)
                    if model_name in ["L3M", "L6M"]:
                        preds_forecast = model_func(train_data_forecast, future_dates)
                    elif "direct" in model_func.__name__:
                        preds_forecast, f_importance = model_func(train_data_forecast, len(future_dates), train_cutoff=forecast_cutoff, full_df=df, key_name=key,key_col=key_col,target_col=target_col,date_col=date_col,**tuned_params.get(model_name.replace('_default',''), {}))
                        if f_importance: all_feature_importances.append({'key': key, 'model': model_name, **f_importance})
                    elif model_name in ["TBF", "LY_trend"]:
                        preds_forecast = model_func(train_data_forecast, len(future_dates), train_cutoff=forecast_cutoff,full_df=df,key_name=key,key_col=key_col,target_col=target_col,date_col=date_col,ma_order=9, rolling_window=6)
                    else:
                        preds_forecast = model_func(train_data_forecast, len(future_dates))
                        
                    raw_forecast_series = pd.Series(np.ravel(preds_forecast), index=future_dates) if preds_forecast is not None else pd.Series(dtype=float, index=future_dates)
                    final_forecasts[model_name] = raw_forecast_series
                    
                    base_df.loc[base_df["date"].isin(future_dates), f"{model_name}_pred"] = raw_forecast_series.values
                    adj_forecast = calculate_bias_adjustment(base_df.loc[base_df["date"].isin(validation_dates), "actual_value"].dropna(), val_preds_dict.get(model_name, pd.Series(dtype=float)), raw_forecast_series)
                    base_df.loc[base_df["date"].isin(future_dates), f"{model_name}_adj_pred"] = adj_forecast.values
                except Exception as e:
                    logger.warning(f"  EXCEPTION during forecast for {model_name} for key {key}: {e!r}")
            
            bands = build_monthly_upper_lower_band(base_df, key_col=key_col, date_col="date", value_col="actual_value", train_till=forecast_cutoff)
            fcst_A_raw = final_forecasts.get(best_model_raw, pd.Series(np.nan, index=future_dates))
            val_actuals = base_df.loc[base_df["date"].isin(validation_dates), "actual_value"].dropna()
            fcst_A_adj = calculate_bias_adjustment(val_actuals, val_preds_dict.get(best_model_raw, pd.Series(dtype=float)), fcst_A_raw)
            fcst_B_adj = calculate_bias_adjustment(val_actuals, val_preds_dict.get(best_model_bias_adj, pd.Series(dtype=float)), final_forecasts.get(best_model_bias_adj, pd.Series(np.nan, index=future_dates)))

            if not bands.empty:
                bands_indexed = bands.set_index("date")
                future_bands = bands_indexed.loc[bands_indexed.index.isin(future_dates)]
                if not future_bands.empty:
                    fcst_A_raw = fcst_A_raw.clip(lower=future_bands["lower_band"], upper=future_bands["upper_band"])
                    fcst_A_adj = fcst_A_adj.clip(lower=future_bands["lower_band"], upper=future_bands["upper_band"])
                    fcst_B_adj = fcst_B_adj.clip(lower=future_bands["lower_band"], upper=future_bands["upper_band"])

            future_mask = base_df["date"].isin(future_dates)
            base_df.loc[future_mask, "fcst_best_raw_model_unadjusted"] = fcst_A_raw.values
            base_df.loc[future_mask, "fcst_best_raw_model_adjusted"] = fcst_A_adj.values
            base_df.loc[future_mask, "fcst_best_adj_model_adjusted"] = fcst_B_adj.values
            base_df['best_model_raw'] = best_model_raw
            base_df['best_model_bias_adj'] = best_model_bias_adj
            
            val_mask, test_mask = base_df["date"].isin(validation_dates), base_df["date"].isin(test_dates)
            base_df.loc[val_mask, 'fcst_best_raw_model_unadjusted'] = base_df.loc[val_mask, f"{best_model_raw}_pred"].values
            base_df.loc[test_mask, 'fcst_best_raw_model_unadjusted'] = base_df.loc[test_mask, f"{best_model_raw}_pred"].values
            base_df.loc[val_mask, 'fcst_best_raw_model_adjusted'] = base_df.loc[val_mask, f"{best_model_raw}_pred"].values
            base_df.loc[test_mask, 'fcst_best_raw_model_adjusted'] = base_df.loc[test_mask, f"{best_model_raw}_adj_pred"].values
            base_df.loc[val_mask, 'fcst_best_adj_model_adjusted'] = base_df.loc[val_mask, f"{best_model_bias_adj}_pred"].values
            base_df.loc[test_mask, 'fcst_best_adj_model_adjusted'] = base_df.loc[test_mask, f"{best_model_bias_adj}_adj_pred"].values

            base_df['period'] = np.select([val_mask, test_mask, future_mask], ['validation', 'testing', 'forecasting'], default='train')

            all_results.append(base_df)
            logger.info(f"--- Finished processing Key: {key} ---")
            # file_handler.flush()

        except Exception as e:
            # <--- ADDED LOGGING: Catch any critical, unhandled errors for a key --->
            logger.error(f"!!! CRITICAL ERROR processing key {key}. Skipping to next key. Error: {e}", exc_info=True)
            # file_handler.flush()
            continue # Move to the next key
        # feature_cols_to_add defined at top of function

    if all_results:
        processed_results = []
        # Loop through results for each key to generate and merge features
        for key_df in all_results:
            key = key_df[key_col].iloc[0]
            
            # Create a series from the actual values to generate features
            # .fillna(0) is important here to handle the forecast period where actuals are NaN
            series = key_df.set_index('date')['actual_value'].fillna(0).copy().rename('target')
            
            # 1. Generate trend and seasonality features
            # Use forecast_cutoff to train the final feature model on all available data
            feature_df = add_trend_seasonality(series, train_only_cutoff=forecast_cutoff)
            
            # 2. Generate lag and rolling window features
            for lag in [1, 2, 3]:
                feature_df[f"lag{lag}"] = feature_df["target"].shift(lag)
            feature_df["rolling_mean_3m"] = feature_df["target"].shift(1).rolling(3).mean()
            feature_df["rolling_mean_6m"] = feature_df["target"].shift(1).rolling(6).mean()
            feature_df["rolling_std_3m"] = feature_df["target"].shift(1).rolling(3).std()
            
            # 3. Generate calendar features
            feature_df["year"] = feature_df.index.year
            feature_df["month"] = feature_df.index.month
            feature_df["quarter"] = feature_df.index.quarter
            
            # 4. Generate monthly ratio feature
            # Use validation_cutoff to be consistent with how ratios were created for model training
            monthly_ratios = get_monthly_ratios_for_cutoff(df, key, validation_cutoff, key_col, target_col, date_col)
            if not monthly_ratios.empty:
                month_to_ratio = monthly_ratios.set_index('month')['avg_monthly_ratio'].to_dict()
                feature_df["monthly_ratio"] = feature_df["month"].map(month_to_ratio).fillna(1/12)
            else:
                feature_df["monthly_ratio"] = 1/12 # Default if no ratios exist
            
            # 5. Merge features back into the key's result DataFrame
            feature_df.drop(columns=['target'], inplace=True)
            updated_key_df = key_df.merge(feature_df[feature_cols_to_add].reset_index(), on='date', how='left')
            # # --- Safety check: ensure feature_cols_to_add is defined ---
            # if 'feature_cols_to_add' not in locals(): #avi1
            #     if 'feature_df' in locals():
            #         feature_cols_to_add = [c for c in feature_df.columns if c not in ['date']]
            #     else:
            #         feature_cols_to_add = []   
            # try:        
            #     updated_key_df = key_df.merge(feature_df[feature_cols_to_add].reset_index(), on='date', how='left')
            # except Exception as e:
            #     logger.warning(f"⚠️ Feature merge failed for key {key}: {e}")
            #     updated_key_df = key_df.copy()   #avi1
                
            processed_results.append(updated_key_df)

        final_results = pd.concat(processed_results, ignore_index=True)
    else:
        # If no results, create an empty DataFrame with the feature columns as well
        # Get all column names from a potential result to build the empty frame
        logger.warning("No results were generated. Returning an empty DataFrame.")
        base_cols = ["date", key_col, seasonal_col, hist_range_col, "actual_value", "period"]
        model_pred_cols = [f"{m}_pred" for m in all_model_names] + [f"{m}_adj_pred" for m in all_model_names]
        forecast_cols = ["fcst_best_raw_model_unadjusted", "fcst_best_raw_model_adjusted", "fcst_best_adj_model_adjusted",
                          "best_model_raw", "best_model_bias_adj"]
        # ✅ FIX: ensure feature_cols_to_add is defined even if skipped in logic above
        if 'feature_cols_to_add' not in locals():
            feature_cols_to_add = []
        empty_cols = base_cols + hierarchy_cols + model_pred_cols + forecast_cols + feature_cols_to_add    #avi3
        #empty_cols = base_cols + hierarchy_cols + model_pred_cols + forecast_cols
        final_results = pd.DataFrame(columns=empty_cols)
        feature_importance_df = pd.DataFrame()
        results = pd.DataFrame(columns=['date', 'key', 'seasonal', 'hist_range', 'actual_value',
                                       'fcst_best_raw_model_unadjusted', 'fcst_best_raw_model_adjusted', 
                                       'fcst_best_adj_model_adjusted', 'best_model_raw', 
                                       'best_model_bias_adj', 'period'])
        return results, final_results, feature_importance_df

    if not all_feature_importances:
        feature_importance_df = pd.DataFrame()
    else:
        fi_df = pd.DataFrame(all_feature_importances)
        feature_importance_df = fi_df.set_index(['key', 'model']).unstack(level='model')        

    results = final_results[['date', 'key', 'seasonal', 'hist_range', 'actual_value',
         'fcst_best_raw_model_unadjusted', 'fcst_best_raw_model_adjusted', 'fcst_best_adj_model_adjusted', 
         'best_model_raw','best_model_bias_adj', 'period'
         ]].copy()
    
    results = results.sort_values([ 'date', key_col]).reset_index(drop=True)

    # results.to_csv(os.path.join(notebook_path,f"{EXPERIMENT_NUMBER}_pipeline_output.csv"), index=False)
    # # Save detailed pipeline output to CSV
    # final_results.to_csv(os.path.join(notebook_path, f"{EXPERIMENT_NUMBER}_detailed_pipeline_output.csv"), index=False)
    # feature_importance_df.to_excel(os.path.join(notebook_path, f"{EXPERIMENT_NUMBER}_feature_importances.xlsx"), index=True)

    return results, final_results, feature_importance_df