import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

model = tf.keras.models.load_model('rgb_classifier_model.h5')
label_classes = np.load('label_classes.npy')

X = []
y_true = []
points_info = []

xml_files = [f for f in os.listdir() if f.endswith('.xml')]

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image file name from XML
        image_tag = root.find(".//image")
        if image_tag is None or "name" not in image_tag.attrib:
            print(f"⚠️ Skipping {xml_file} (missing image name)")
            continue

        image_file = image_tag.attrib['name']
        if not os.path.exists(image_file):
            print(f"⚠️ Skipping {xml_file} (image {image_file} not found)")
            continue

        image = Image.open(image_file)
        pixels = np.array(image)

        for points_tag in root.findall(".//image/points"):
            label = points_tag.attrib['label']
            coords = points_tag.attrib['points'].split(';')
            for point in coords:
                x, y = map(lambda v: int(float(v)), point.split(','))
                if 0 <= y < pixels.shape[0] and 0 <= x < pixels.shape[1]:
                    rgb = pixels[y, x, :3]
                    X.append(rgb)
                    y_true.append(label)
                    points_info.append((image_file, x, y, label))

    except Exception as e:
        print(f"❌ Error processing {xml_file}: {e}")

if len(X) == 0:
    print("❌ No points to evaluate.")
    exit()

X = np.array(X)
predictions = model.predict(X)
predicted_indices = np.argmax(predictions, axis=1)
y_pred = label_classes[predicted_indices]

correct = 0
for i, (img, x, y, true_label) in enumerate(points_info):
    pred_label = y_pred[i]
    status = "✅ Correct" if pred_label == true_label else f"❌ Wrong (Predicted: {pred_label})"
    print(f"[{img}] Point ({x},{y}) → Label: {true_label} → {status}")
    if pred_label == true_label:
        correct += 1

accuracy = correct / len(y_true)
print(f"\n📊 Overall Accuracy on all images: {accuracy * 100:.2f}%")

