import xml.etree.cElementTree as ET
import os

'''Modify the category name of the xml file to a specific name'''

def modify_one_xml(xml_path):
    file = open(xml_path, encoding='UTF-8')
    tree = ET.parse(file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls_name = obj.find('name')
        if cls_name.text == '吊车':
            cls_name.text = "DiaoChe"
        elif cls_name.text == '挖掘机' or cls_name.text == '铲车':
            cls_name.text = "ShiGongJiXie"
        elif cls_name.text == '导线异物':
            cls_name.text = "DaoXianYiWu"
        elif cls_name.text == '塔吊':
            cls_name.text = "TaDiao"
        elif cls_name.text == '烟雾':
            cls_name.text = "YanHuo"
        elif cls_name.text == '杆塔':
            cls_name.text = "GanTa"
        elif cls_name.text == '塑料':
            cls_name.text = "SuLiao"
        elif cls_name.text == '支撑件':
            cls_name.text = "ZhiChengJian"
        elif cls_name.text == '混凝土搅拌机':
            cls_name.text = "HunNingTuJiaoBanJi"
        elif cls_name.text == '绝缘子':
            cls_name.text = "JueYuanZi"
        elif cls_name.text == '压路机':
            cls_name.text = "YaLuJi"
        elif cls_name.text == '支撑杆':
            cls_name.text = "ZhiChengGan"
        else:
            pass
    tree.write(xml_path)

path = '/shared/xjd/DataSets/transmission_line_detection/self_labeled_xml'
filelist = os.listdir(path)
xml_name_list = [filename for filename in filelist if filename[-3:] == 'xml']
print(len(xml_name_list))
for xml_name in xml_name_list:
    modify_one_xml(os.path.join(path, xml_name))

print('Finished!')