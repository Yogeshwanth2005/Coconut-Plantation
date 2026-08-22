import os
import json
from glob import glob
from collections import defaultdict

folder_path = "geojson_files"

# { region: set of (width, height) }
region_patch_sizes = defaultdict(set)

def get_region_name(filename):
    parts = filename.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

for file_path in glob(os.path.join(folder_path, "*.geojson")):
    file_name = os.path.basename(file_path)
    region = get_region_name(file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom.get("coordinates", [])[0]  # first ring
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            width = round(max(xs) - min(xs), 2)
            height = round(max(ys) - min(ys), 2)
            region_patch_sizes[region].add((width, height))

# Print results
for region, sizes in region_patch_sizes.items():
    print(f"\nRegion: {region}")
    for w, h in sorted(sizes):
        print(f"  Patch size: {w} × {h} px")

# Save to CSV
import pandas as pd
rows = []
for region, sizes in region_patch_sizes.items():
    for w, h in sorted(sizes):
        rows.append({"Region": region, "Patch Width (px)": w, "Patch Height (px)": h})

df = pd.DataFrame(rows)
output_path = "region_patch_sizes.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\n✅ Patch sizes saved to {output_path}")
