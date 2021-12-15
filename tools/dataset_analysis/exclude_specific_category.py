import xml.etree.cElementTree as ET
import os
import shutil

def exclude_specific_category(xml_path, xml_name, img_save_path, xml_save_path):
    xml_path = os.path.join(xml_path, xml_name)
    image_path = os.path.join(img_path, xml_name[:-3] + 'jpg')
    file = open(xml_path)
    tree = ET.parse(file)
    root = tree.getroot()

    log = False
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name == 'negative' or cls_name == 'daoxian':
            log = True

    file.close()
    if log:
        shutil.move(image_path, img_save_path)
        shutil.move(xml_path, xml_save_path)



xml_path = '/shared/xjd/DataSets/transmission_line_detection/train14000/Annotations'
img_path = '/shared/xjd/DataSets/transmission_line_detection/train14000/JPEGImages'
xml_save_path = '/shared/xjd/DataSets/transmission_line_detection/train14000/exclude_Annotations'
img_save_path = '/shared/xjd/DataSets/transmission_line_detection/train14000/exclude_JPEGImages'
filelist = os.listdir(xml_path)
xml_name_list = [filename for filename in filelist if filename[-3:] == 'xml']
print(len(xml_name_list))

if not os.path.exists(xml_save_path):
    os.makedirs(xml_save_path)
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

for xml_name in xml_name_list:
    exclude_specific_category(xml_path, xml_name, img_save_path, xml_save_path)

print('Finished!')