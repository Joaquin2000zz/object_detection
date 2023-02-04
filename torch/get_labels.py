import xml.etree.ElementTree as ET

def xml_to_dict(xml_path):
    """
    - Decode the .xml file
    Returns: the labels.
        the image size, object label and bounding box
        coordinates altogether with the filename as a dictionary.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {"filename": xml_path,
            "image_width": int(root.find("./size/width").text),
            "image_height": int(root.find("./size/height").text),
            "image_channels": int(root.find("./size/depth").text),
            "label": root.find("./object/name").text,
            "x1": int(root.find("./object/bndbox/xmin").text),
            "y1": int(root.find("./object/bndbox/ymin").text),
            "x2": int(root.find("./object/bndbox/xmax").text),
            "y2": int(root.find("./object/bndbox/ymax").text)}