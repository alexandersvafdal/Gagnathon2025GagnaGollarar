import pandas as pd

# Read both CSV files
al_total = pd.read_csv('data/cleaned/AL_TOTAL.csv')
electricity = pd.read_csv('data/aukagogn/IcelandElectrisityPrice.csv')

# Merge the dataframes on matching dates
merged = al_total.merge(
    electricity, 
    left_on='time', 
    right_on='observation_date', 
    how='left'
)

# Drop the redundant observation_date column
merged = merged.drop('observation_date', axis=1)

# Reorder columns to put CP0451ISM086NEST after the existing columns
cols = ['time', 'DXY_dollar_index', 'Raungengi_Index', 'CP0451ISM086NEST']
merged = merged[cols]

# Save the merged data to a new file
output_file = 'data/cleaned/AL_TOTAL_updated.csv'
merged.to_csv(output_file, index=False)

print(f'Successfully merged CP0451ISM086NEST column into {output_file}')
print(f'Total rows: {len(merged)}')
print(f'Non-null CP0451ISM086NEST values: {merged["CP0451ISM086NEST"].notna().sum()}')
print('\nFirst few rows:')
print(merged.head(10))
