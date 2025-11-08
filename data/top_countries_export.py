import pandas as pd

# Load your dataset
df = pd.read_parquet('./data/Exports-by-branches-of-processing-and-countries-2015-2025.parquet')

# Convert DATA column to numeric (force bad values to NaN)
df["DATA"] = pd.to_numeric(df["DATA"], errors="coerce")

# Filter for FOB value rows
fob = df[df["Unit"] == "Fob value"]

# Group by country and sum FOB value
top_importers = (
    fob.groupby("Country")["DATA"]
       .sum()
       .sort_values(ascending=False)
)

# Display top 20 importing countries
print("\nTop 20 countries by FOB value:\n")
print(top_importers.head(20))
