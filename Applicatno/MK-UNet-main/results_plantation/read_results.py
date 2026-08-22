"""Quick script to read and display test results from the Excel file."""
import pandas as pd
import sys

xlsx = r'd:\Project\Coconut Plantation\Applicatno\MK-UNet-main\results_plantation\Results_Plantation_MK_UNet_ShallowDec_L_bs8_lr0.0005_e10_cls32_run1_t105047_Plantation_test.xlsx'

df = pd.read_excel(xlsx)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)

# Save as CSV for easy reading
csv_path = xlsx.replace('.xlsx', '.csv')
df.to_csv(csv_path, index=False)
print(f"Saved CSV to: {csv_path}")

# Print summary
print(f"\nShape: {df.shape}")
print(f"\n--- AVERAGE ROW ---")
avg = df[df['Name'] == 'AVERAGE'].iloc[0]
for col, val in avg.items():
    if col == 'Name':
        continue
    print(f"  {col}: {val}")
