import os
import xml.etree.ElementTree as ET

# -------- CONFIG --------
input_folder = "annotations1"       # folder containing original XML files
output_folder = "annotations1"   # folder to store scaled XML files
x = 2   # scale factor (2 = 1/2, 4 = 1/4, 8 = 1/8)
# ------------------------

scale = 1 / x

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.endswith(".xml"):

        input_path = os.path.join(input_folder, filename)

        # Save with SAME name in output folder
        output_path = os.path.join(output_folder, filename)

        tree = ET.parse(input_path)
        root = tree.getroot()

        for image in root.iter("image"):

            original_width = float(image.get("width"))
            original_height = float(image.get("height"))

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            image.set("width", str(new_width))
            image.set("height", str(new_height))

            for points in image.iter("points"):

                point_string = points.get("points")
                point_list = point_string.split(";")

                new_points = []

                for p in point_list:
                    x_old, y_old = map(float, p.split(","))

                    x_new = x_old * scale
                    y_new = y_old * scale

                    new_points.append(f"{x_new},{y_new}")

                points.set("points", ";".join(new_points))

        tree.write(output_path)

print("All XML files processed and saved with original names.")