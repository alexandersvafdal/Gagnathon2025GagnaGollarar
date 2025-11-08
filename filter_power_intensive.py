import pandas as pd
import re

# Read the CSV file
df = pd.read_csv('data/Exports_imports_by_US_2015_2025.csv', sep=';', encoding='utf-8')

# Find the row with '71 Products of power intensive plants'
branch_row = df[df.iloc[:, 0].str.contains('71 Products of power intensive plants', na=False)]

if branch_row.empty:
    print("Branch not found!")
    exit(1)

# Get the data from this row
data_row = branch_row.iloc[0]

# Get the header row (row with months)
header_row = df.iloc[2]

# Create list to store results
results = []

# Process each month column
for col_idx in range(3, len(header_row)):
    month_str = header_row.iloc[col_idx]
    value = data_row.iloc[col_idx]
    
    if pd.notna(month_str) and pd.notna(value) and value != '' and value != '0,00' and value != 0:
        # Parse month (format: 2015M01)
        match = re.match(r'(\d{4})M(\d{2})', str(month_str))
        if match:
            year = match.group(1)
            month = match.group(2)
            date_str = f"{year}-{month}-01"
            
            # Clean value (replace comma with period for decimal)
            value_clean = str(value).replace(',', '.')
            try:
                value_float = float(value_clean)
                if value_float != 0:
                    results.append({
                        'Month': date_str,
                        'Branches': '71 Products of power intensive plants',
                        'DATA': value_float
                    })
            except ValueError:
                continue

# Create DataFrame
result_df = pd.DataFrame(results)

# Save to new file
output_file = 'data/power_intensive_plants_filtered.csv'
result_df.to_csv(output_file, index=False)

print(f"Successfully created {output_file}")
print(f"Total rows: {len(result_df)}")
print(f"\nFirst 10 rows:")
print(result_df.head(10))
print(f"\nLast 10 rows:")
print(result_df.tail(10))
