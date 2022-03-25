# import xml.etree.cElementTree as ET
# import os

# classes = []
# def gen_classes(path, xml_file):
#     file = open(os.path.join(path, xml_file), encoding='UTF-8')
#     tree = ET.parse(file)
#     root = tree.getroot()
#     for obj in root.iter('object'):
#         cls_name = obj.find('name').text
#         if cls_name in classes:
#             pass
#         else:
#             classes.append(cls_name)
#     return classes

# path = '/data/DataSets/transmission_line_detection/train14000_xml'

# xml_files = os.listdir(path)
# for xml_file in xml_files:
#     gen_classes(path, xml_file)

# print(classes)

import xml.etree.cElementTree as ET
import os

# 1.首先需要查看一下所有xml文件中的类别都有哪些?
def gen_classes(xml_path):
    xml_files = os.listdir(xml_path)
    classes = []
    for xml_file in xml_files:
        file = open(os.path.join(xml_path, xml_file), encoding='UTF-8')
        tree = ET.parse(file)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name in classes:
                pass
            else:
                classes.append(cls_name)
    
    return classes

xml_path = '/shared/xjd/DataSets/transmission_line_detection/train14000_xml'
classes = gen_classes(xml_path)
print(classes)