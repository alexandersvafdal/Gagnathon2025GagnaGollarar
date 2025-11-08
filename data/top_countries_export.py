# import pandas as pd

# # Load your dataset
# df = pd.read_parquet('./data/Exports-by-branches-of-processing-and-countries-2015-2025.parquet')

# # Convert DATA column to numeric (force bad values to NaN)
# df["DATA"] = pd.to_numeric(df["DATA"], errors="coerce")

# # Filter for FOB value rows
# fob = df[df["Unit"] == "Fob value"]

# # Group by country and sum FOB value
# top_importers = (
#     fob.groupby("Country")["DATA"]
#        .sum()
#        .sort_values(ascending=False)
# )

# # Display top 20 importing countries
# print("\nTop 20 countries by FOB value:\n")
# print(top_importers.head(20))

import pandas as pd

# Load your dataset
df = pd.read_parquet('./data/Exports-by-branches-of-processing-and-countries-2015-2025.parquet')

df["DATA"] = pd.to_numeric(df["DATA"], errors="coerce")

# Extract year from Month column (e.g. 2015M01 → 2015)
df["Year"] = df["Month"].astype(str).str[:4].astype(int)

# Filter FOB only
fob = df[df["Unit"] == "Fob value"]

# Compute yearly averages
yearly_avg = (
    fob.groupby(["Country", "Year"])["DATA"]
        .mean()
        .reset_index()
        .sort_values(["Country", "Year"])
)

print(yearly_avg.head(20))

avg_yearly_by_country = (
    yearly_avg.groupby("Country")["DATA"]
              .mean()
              .sort_values(ascending=False)
)

print(avg_yearly_by_country.head(20))
