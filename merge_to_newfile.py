import pandas as pd

# Read both CSV files as strings to preserve formatting
al_total = pd.read_csv('data/cleaned/AL_TOTAL.csv', dtype=str)
electricity = pd.read_csv('data/aukagogn/IcelandElectrisityPrice.csv', dtype=str)

# Merge on dates: AL_TOTAL.time == IcelandElectrisityPrice.observation_date
merged = al_total.merge(
    electricity,
    left_on='time',
    right_on='observation_date',
    how='left'
)

# Drop the redundant observation_date column if present
if 'observation_date' in merged.columns:
    merged = merged.drop(columns=['observation_date'])

# Ensure the month format is yyyy-MM-dd. The `time` column already appears in that format in source files.
# If needed, we could parse and format, but we'll assume source format is correct per attachments.

# Write to a new file to avoid permission errors on the original
out_path = 'data/cleaned/AL_TOTAL_merged.csv'
merged.to_csv(out_path, index=False)

print(f'Wrote: {out_path}')
print(f'Total rows: {len(merged)}')
if 'CP0451ISM086NEST' in merged.columns:
    print('Non-null CP0451ISM086NEST count:', merged['CP0451ISM086NEST'].notna().sum())
else:
    print('Column CP0451ISM086NEST not found in merged output')
