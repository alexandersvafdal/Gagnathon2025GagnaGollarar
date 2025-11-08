import pandas as pd

# Read both CSV files
al_total = pd.read_csv('data/cleaned/AL_TOTAL_updated.csv')
exports = pd.read_csv('data/Exports-by-branches-of-processing-and-countries-2015-2025-onlyAL_updated.csv')

print(f"AL_TOTAL shape: {al_total.shape}")
print(f"Exports data shape: {exports.shape}")
print(f"\nAL_TOTAL columns: {list(al_total.columns)}")
print(f"Exports columns: {list(exports.columns)}")

# Keep only the columns we need from exports (including DATA)
exports_filtered = exports[['Month', 'Branches', 'Country', 'Unit', 'DATA']].copy()

# Merge with AL_TOTAL on Month = time
merged = al_total.merge(
    exports_filtered,
    left_on='time',
    right_on='Month',
    how='left'
)

# Drop the duplicate Month column
if 'Month' in merged.columns:
    merged = merged.drop('Month', axis=1)

# Reorder columns to match requested order
merged = merged[['time', 'DXY_dollar_index', 'Raungengi_Index', 'CP0451ISM086NEST', 'Branches', 'Country', 'Unit', 'DATA']]

# Save to new file
output_file = 'data/cleaned/AL_TOTAL_v2.csv'
merged.to_csv(output_file, index=False)

print(f"\nSuccessfully created {output_file}")
print(f"Final shape: {merged.shape}")
print(f"Total columns: {len(merged.columns)}")
print(f"\nFirst few column names:")
print(list(merged.columns)[:20])
print(f"\nFirst 5 rows of merged data:")
print(merged.head())
