import json
import glob
import statistics
from collections import defaultdict
import webcolors
from webcolors import CSS3_NAMES_TO_HEX
import os
import csv

folder_path = "geojson_files/*.geojson"

# Store counts and file names per color
annotations_by_color = defaultdict(lambda: {"counts": [], "files": set(), "rgbs": set()})
region_to_all_files = defaultdict(set)

def rgb_to_name(rgb_tuple):
    """Convert an RGB tuple to the nearest CSS3 color name."""
    try:
        return webcolors.rgb_to_name(rgb_tuple, spec='css3').title()
    except ValueError:
        # ✅ Invert CSS3_NAMES_TO_HEX manually
        css3_hex_to_names = {v: k for k, v in CSS3_NAMES_TO_HEX.items()}
        min_distance = float('inf')
        closest_name = None
        for hex_value, name in css3_hex_to_names.items():
            r, g, b = webcolors.hex_to_rgb(hex_value)
            distance = (r - rgb_tuple[0]) ** 2 + (g - rgb_tuple[1]) ** 2 + (b - rgb_tuple[2]) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name.title()

def get_region_name(filename):
    """Assume region name is before first underscore."""
    return filename.split("_")[0]

# Pass 1: collect all files per region
for filepath in glob.glob(folder_path):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    region = get_region_name(filename)
    region_to_all_files[region].add(filename)

# Pass 2: collect annotation stats
for filepath in glob.glob(folder_path):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.splitext(os.path.basename(filepath))[0]
    region = get_region_name(filename)

    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})

        if geom.get("type") == "MultiPoint":
            color = props.get("color") or props.get("stroke") or props.get("fill")
            rgb_tuple = None

            if isinstance(color, list) and len(color) == 3:
                rgb_tuple = tuple(color)
                color_name = rgb_to_name(rgb_tuple)
            elif isinstance(color, tuple) and len(color) == 3:
                rgb_tuple = color
                color_name = rgb_to_name(rgb_tuple)
            elif color is None:
                color_name = "No Color"
            else:
                try:
                    rgb_tuple = webcolors.hex_to_rgb(color)
                    color_name = rgb_to_name(rgb_tuple)
                except ValueError:
                    color_name = str(color)

            if rgb_tuple and color_name == "No Color":
                color_name = f"No Color (RGB: {rgb_tuple})"

            num_points = len(geom.get("coordinates", []))
            annotations_by_color[color_name]["counts"].append(num_points)
            annotations_by_color[color_name]["files"].add(filename)
            if rgb_tuple:
                annotations_by_color[color_name]["rgbs"].add(rgb_tuple)

# Prepare CSV data
csv_rows = []
csv_rows.append(["Color", "Original RGB(s)", "Number of Annotations", "Mean", "SD", "SD/Mean", "Found In"])

for color, data_dict in annotations_by_color.items():
    counts = data_dict["counts"]
    files_with_color = data_dict["files"]
    rgbs = ", ".join(map(str, sorted(data_dict["rgbs"]))) if data_dict["rgbs"] else ""

    mean_val = statistics.mean(counts)
    std_val = statistics.stdev(counts) if len(counts) > 1 else 0.0
    ratio = (std_val / mean_val) if mean_val != 0 else 0

    # Group files by region
    region_to_files_with_color = defaultdict(set)
    for fname in files_with_color:
        region_to_files_with_color[get_region_name(fname)].add(fname)

    display_list = []
    for region, all_files in region_to_all_files.items():
        if region in region_to_files_with_color:
            if region_to_files_with_color[region] == all_files:
                display_list.append(region)
            else:
                display_list.extend(sorted(region_to_files_with_color[region])) 

    csv_rows.append([
        color,
        rgbs,
        len(counts),
        f"{mean_val:.2f}",
        f"{std_val:.2f}",
        f"{ratio:.4f}",
        ", ".join(sorted(display_list))
    ])

# Save CSV
output_csv = "multi_point_annotation_stats.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f" CSV file saved as {output_csv}")
