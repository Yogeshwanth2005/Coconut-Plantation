import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

TARGET_SIZE = (256, 256)

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "green-400": 2,
    "non-green-400": 3,
    "green-300": 4,
    "non-green-300": 5,
    "Coconut-300": 6,
    "Cocount-400": 7,
    "Coconut-800": 8
}

def extract_point_features(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    features = []
    labels = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, row['filename'])
        img = cv2.imread(image_path)
        if img is None:
            continue

        point_strs = row['points'].split(';')
        for point_str in point_strs:
            x, y = map(float, point_str.strip().split(','))
            x, y = int(x), int(y)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                rgb = img[y, x] / 255.0
                features.append(rgb)
                labels.append(LABEL_MAP.get(row['label'], -1))

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def build_simple_classifier(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    features, labels = extract_point_features('all_annotations.csv', 'images')
    mask = labels >= 0
    features, labels = features[mask], labels[mask]

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = build_simple_classifier(input_shape=(3,), num_classes=len(LABEL_MAP))
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

    os.makedirs('model', exist_ok=True)
    model.save('model/point_classifier.h5')
    print("Model saved to model/point_classifier.h5")

if __name__ == '__main__':
    main()
