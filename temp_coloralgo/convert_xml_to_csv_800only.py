# convert_xml_to_csv2.py
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_data = []
    for image_tag in root.findall('image'):
        filename = image_tag.attrib.get('name')
        width = int(image_tag.attrib.get('width', 0))
        height = int(image_tag.attrib.get('height', 0))

        for point_tag in image_tag.findall('points'):
            label = point_tag.attrib.get('label')
            if '800' not in label:
                continue

            points = point_tag.attrib.get('points')
            all_data.append({
                'filename': filename,
                'width': width,
                'height': height,
                'label': label,
                'points': points
            })

    return all_data

def convert_all_xmls(annotations_folder):
    all_entries = []
    xml_files = glob.glob(os.path.join(annotations_folder, '*.xml'))
    for xml_file in xml_files:
        entries = parse_xml_file(xml_file)
        all_entries.extend(entries)

    df = pd.DataFrame(all_entries)
    return df

if __name__ == "__main__":
    annotations_folder = "annotations"
    output_csv = "all_annotations.csv"

    df = convert_all_xmls(annotations_folder)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} rows to {output_csv}")
