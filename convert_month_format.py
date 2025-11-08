import pandas as pd

# Read the CSV file
df = pd.read_csv('data/Exports-by-branches-of-processing-and-countries-2015-2025-onlyAL.csv')

print(f"Original data shape: {df.shape}")
print(f"Sample of original Month values:")
print(df['Month'].head(10))

# Convert Month column from 2015M01 format to 2015-01-01 format
df['Month'] = pd.to_datetime(df['Month'], format='%YM%m').dt.strftime('%Y-%m-01')

print(f"\nSample of converted Month values:")
print(df['Month'].head(10))

# Save to a new file
output_file = 'data/Exports-by-branches-of-processing-and-countries-2015-2025-onlyAL_updated.csv'
df.to_csv(output_file, index=False)

print(f"\nSuccessfully updated Month format to yyyy-MM-01")
print(f"Total rows: {len(df)}")
print(f"Saved to: {output_file}")
