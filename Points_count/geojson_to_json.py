import os
import json

# Input and output folders
input_folder = r"D:\Project\Points_count\geojson_files"
output_folder = r"D:\Project\Points_count\json_files"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all .geojson files
for filename in os.listdir(input_folder):
    if filename.endswith(".geojson"):
        geojson_path = os.path.join(input_folder, filename)

        try:
            # Read the GeoJSON
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if the file content is a dictionary or list
            if not isinstance(data, (dict, list)):
                raise ValueError(f"{filename} is not a valid GeoJSON (found {type(data)})")

            # Create new filename with .json extension
            json_filename = filename.replace(".geojson", ".json")
            json_path = os.path.join(output_folder, json_filename)

            # Save as JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            print(f"✅ Converted: {filename} → {json_filename}")

        except json.JSONDecodeError as e:
            print(f"❌ Error decoding {filename}: {e}")
        except Exception as e:
            print(f"⚠️ Skipped {filename} due to error: {e}")

print("🎉 Conversion complete!")
