# test_pipeline_run.py
import pandas as pd
from forecasting_pipeline import forecast_pipeline_debug, add_hist_range
from main import _prepare_df

df = pd.read_csv("C:/Users/aksha/Downloads/API_Input_test.csv")
# print("✅ df completed")
# df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df.csv", index=False)
#print(df)
params = {
    "date_col": "sales_date",
    "target_col": "sales",
    "key_col": "key",
    "seasonal_col": "seasonal",
    "hist_range_col": "hist_range",
    "key_components": "business_type x store_type x region x subcat x product_code"
}

# print("✅ df completed2")
# df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df2.csv", index=False)
df_prepared = _prepare_df(df, params)
df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/debug_prepared_df.csv", index=False)

# print("✅ df completed")
# print(df)
#print("✅ forecast_direct.csv generated successfully!")
print("✅ df completed")
df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/input_df3.csv", index=False)
print(" df_prepared prepared successfully!")
print(df_prepared)

# result = forecast_pipeline_debug(
#     df=df_prepared,
#     parameters=params,
#     validation_cutoff="2024-04-30",
#     test_cutoff="2024-07-31",
#     forecast_cutoff="2024-12-31",
#     forecasting_horizon=3,
#     debug_keys=None
# )

# If forecast_pipeline_debug returns tuple
if isinstance(result, tuple):
    result_df = result[0]
else:
    result_df = result

result_df.to_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/forecast_direct.csv", index=False)
print("✅ forecast_direct.csv generated successfully!")


