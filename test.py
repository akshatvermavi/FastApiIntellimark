import pandas as pd

api_out = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/FastApi_output3/forecast_results.csv")
direct_out = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/FastOutputTest/forecast_direct.csv")

# Check for equality
print("Are the two outputs identical?", api_out.equals(direct_out))

# Optional: show differences if not identical
if not api_out.equals(direct_out):
    diff = api_out.compare(direct_out, align_axis=0)
    print("Differences found:\n", diff.head())
