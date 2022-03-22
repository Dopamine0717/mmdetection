import xml.etree.cElementTree as ET
import os

classes = []
def gen_classes(path, xml_file):
    file = open(os.path.join(path, xml_file), encoding='UTF-8')
    tree = ET.parse(file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name in classes:
            pass
        else:
            classes.append(cls_name)
    return classes

path = '/shared/xjd/DataSets/transmission_line_detection/self_labeled_xml'

xml_files = os.listdir(path)
for xml_file in xml_files:
    gen_classes(path, xml_file)

print(classes)