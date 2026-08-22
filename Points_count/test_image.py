import os
import json
import math
import pandas as pd
from collections import defaultdict
from glob import glob

def mean(lst):
    return sum(lst) / len(lst) if lst else 0

def sample_std(lst):
    if len(lst) < 2:
        return 0.0
    avg = mean(lst)
    return math.sqrt(sum((x - avg) ** 2 for x in lst) / (len(lst) - 1))

folder_path = "geojson_files"

# Dictionary: { image_file: { color: [counts] } }
image_data = defaultdict(lambda: defaultdict(list))

for file_path in glob(os.path.join(folder_path, "*.geojson")):
    file_name = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for feature in data.get("features", []):
        if feature.get("geometry", {}).get("type") != "MultiPoint":
            continue

        coords = feature["geometry"]["coordinates"]
        num_points = len(coords)

        props = feature.get("properties", {})
        color = None

        if "color" in props and props["color"]:
            if isinstance(props["color"], list) and len(props["color"]) == 3:
                color = tuple(props["color"])
        elif "stroke" in props and props["stroke"]:
            stroke = props["stroke"].lstrip("#")
            if len(stroke) == 6:
                color = tuple(int(stroke[i:i+2], 16) for i in (0, 2, 4))

        if not color:
            color = "No Color"

        image_data[file_name][color].append(num_points)

# Store results for Excel
rows = []

for image, colors in image_data.items():
    print(f"\nImage: {image}")
    for color, point_counts in colors.items():
        m = mean(point_counts)
        sd = sample_std(point_counts)
        cv = sd / m if m != 0 else 0  # SD/Mean

        # Console output
        print(f"  Color: {color} | Annotations: {len(point_counts)} | "
              f"Mean: {m:.2f} | Std: {sd:.2f} | SD/Mean: {cv:.4f}")

        # Save same data to CSV
        rows.append({
            "Image": image,
            "Color": str(color),
            "Annotations": len(point_counts),
            "Mean": round(m, 2),
            "Std": round(sd, 2),
            "SD/Mean": round(cv, 4)
        })

# Save to CSV
df = pd.DataFrame(rows)
output_path = "image_color_stats.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\n✅ Results saved to {output_path}")
