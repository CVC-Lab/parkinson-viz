#!/usr/bin/env python3
"""
Convert CSV data to JSON for GitHub Pages deployment
Runs the same data loading and processing as the main app,
then exports to JSON for client-side JavaScript consumption
"""

import json
import sys
from data_pipeline import ParkinsonDataLoader

print("=" * 60)
print("Converting Parkinson's Disease Dataset to JSON")
print("=" * 60)

# Load all datasets using the existing loader
loader = ParkinsonDataLoader()
df_merged = loader.load_all_datasets()

if df_merged.empty:
    print("❌ No data loaded. Please check file paths.")
    sys.exit(1)

# Convert to JSON-friendly format
print("\n📊 Converting to JSON format...")
data_dict = df_merged.to_dict('records')

# Replace NaN with None for proper JSON
for record in data_dict:
    for key, value in record.items():
        if isinstance(value, float):
            import math
            if math.isnan(value) or math.isinf(value):
                record[key] = None

# Write to JSON file
output_path = 'data/merged_data.json'
with open(output_path, 'w') as f:
    json.dump(data_dict, f, indent=2)

file_size = len(json.dumps(data_dict)) / 1024 / 1024

print(f"\n✅ Success!")
print(f"   📁 Output: {output_path}")
print(f"   📊 Records: {len(data_dict)}")
print(f"   👥 Patients: {df_merged['PATNO'].nunique()}")
print(f"   💾 Size: {file_size:.2f} MB")
print(f"\n🎉 Data conversion complete!")
print("\nYou can now use this data in the JavaScript web application.")
