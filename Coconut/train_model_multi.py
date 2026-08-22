import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

X_data = []
y_labels = []

xml_files = [f for f in os.listdir() if f.endswith('.xml')]

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image file name from XML content
        image_tag = root.find(".//image")
        if image_tag is None or "name" not in image_tag.attrib:
            print(f"⚠️ Skipping {xml_file} (missing <image name=...>)")
            continue

        image_file = image_tag.attrib['name']
        if not os.path.exists(image_file):
            print(f"⚠️ Skipping {xml_file} (image file {image_file} not found)")
            continue

        # Load image
        image = Image.open(image_file)
        pixels = np.array(image)

        for points_tag in root.findall(".//image/points"):
            label = points_tag.attrib['label']
            coords = points_tag.attrib['points'].split(';')
            for point in coords:
                x, y = map(lambda v: int(float(v)), point.split(','))
                if 0 <= y < pixels.shape[0] and 0 <= x < pixels.shape[1]:
                    rgb = pixels[y, x, :3]
                    X_data.append(rgb)
                    y_labels.append(label)

    except Exception as e:
        print(f"❌ Error processing {xml_file}: {e}")

# Check if data was collected
if len(X_data) == 0:
    print("❌ No valid data found. Please check your XML and image files.")
    exit()

# Convert to NumPy and encode
X_data = np.array(X_data)
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
np.save('label_classes.npy', le.classes_)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1)

model.save('rgb_classifier_model.h5')
print("✅ Training complete. Model and labels saved.")

