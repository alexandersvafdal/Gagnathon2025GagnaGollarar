import pandas as pd

# Read the two files
tonnes_df = pd.read_csv('data/cleaned/AL_TOTAL_v2_Tonnes.csv')
fobvalue_df = pd.read_csv('data/cleaned/AL_TOTAL_v2_fobValue.csv')

# Rename DATA columns
tonnes_df = tonnes_df.rename(columns={'DATA': 'tonnes'})
fobvalue_df = fobvalue_df.rename(columns={'DATA': 'fobvalue'})

# Drop Unit column from both dataframes
tonnes_df = tonnes_df.drop(columns=['Unit'])
fobvalue_df = fobvalue_df.drop(columns=['Unit'])

# Merge on all common columns except the renamed DATA columns
merge_columns = ['time', 'DXY_dollar_index', 'Raungengi_Index', 'CP0451ISM086NEST', 'Branches', 'Country']

merged_df = tonnes_df.merge(
    fobvalue_df,
    on=merge_columns,
    how='outer'
)

# Reorder columns to have tonnes and fobvalue at the end
column_order = merge_columns + ['tonnes', 'fobvalue']
merged_df = merged_df[column_order]

# Save to new file
merged_df.to_csv('data/cleaned/AL_TOTAL_v2_combined.csv', index=False)

print(f"Tonnes shape: {tonnes_df.shape}")
print(f"Fobvalue shape: {fobvalue_df.shape}")
print(f"Combined shape: {merged_df.shape}")
print(f"\nFirst few rows:")
print(merged_df.head())
print(f"\nSaved to: data/cleaned/AL_TOTAL_v2_combined.csv")
