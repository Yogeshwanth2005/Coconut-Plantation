import os, re, json, string
from pathlib import Path

# ---------- CONFIG ----------
INPUT_FOLDER  = "D:\\Project\\Coconut Plantation\\Applicatno\\images"        # folder containing your existing JSON files
OUTPUT_FOLDER = "D:\\Project\\Coconut Plantation\\Applicatno\\annotations"   # folder to save rewritten JSON files
# -----------------------------

def get_region_prefix(file_path: str) -> str:
    """Return first two letters of region name (e.g., Karavatti_800_1.json → KA)."""
    base = os.path.basename(file_path)
    return base[:2].upper() if len(base) >= 2 else "XX"

def get_image_number(file_path: str) -> str:
    """Extract the final number after the last underscore (e.g., 800_1.json → 1)."""
    m = re.search(r"_(\d+)(?=\.[^.]+$)", file_path)
    if m:
        return m.group(1)
    all_nums = re.findall(r"_(\d+)", file_path)
    return all_nums[-1] if all_nums else "0"

def get_range_letter(num_points: int) -> str:
    """Assign range letter based on number of points."""
    if num_points == 0:
        return "Z"   # no points
    idx = (num_points - 1) // 10  # 1–10→A, 11–20→B, etc.
    letters = string.ascii_uppercase
    return letters[idx] if idx < len(letters) else "Z"

def stable_sort(entries):
    """Ensure deterministic ordering for consistent Box IDs."""
    def key_top_left(e):
        tl = e.get("Coordinates", {}).get("Top-left", [0, 0])
        return (tl[1], tl[0])  # sort by y, then x
    return sorted(entries, key=key_top_left)

def process_json(file_path: Path):
    region_prefix = get_region_prefix(file_path.name)
    image_no = get_image_number(file_path.name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{file_path.name} is not a list of annotations.")

    data = stable_sort(data)

    updated_boxes = []
    count_map = {}  # track counts per category

    for entry in data:
        points = entry.get("Points", [])
        num_points = len(points)
        range_letter = get_range_letter(num_points)
        category = f"{region_prefix}-{image_no}-{range_letter}"

        count_map.setdefault(category, 0)
        count_map[category] += 1
        box_id = f"{category}-{count_map[category]}"

        # update entry
        entry["Category"] = category
        entry["Box ID"] = box_id
        updated_boxes.append(entry)

    return updated_boxes

def process_all(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]

    if not files:
        print(f"⚠️ No JSON files found in '{input_folder}'. Nothing to process.")
        return

    for file in files:
        in_path = Path(input_folder) / file
        if not in_path.exists():
            print(f"⚠️ Missing JSON file: {file} — skipping.")
            continue  # <- SKIP if JSON doesn't exist

        try:
            updated = process_json(in_path)
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
            continue

        out_path = Path(output_folder) / file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=4, ensure_ascii=False)

        print(f"✅ {file} processed → saved to {output_folder}")

if __name__ == "__main__":
    process_all(INPUT_FOLDER, OUTPUT_FOLDER)