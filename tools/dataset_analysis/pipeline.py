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

xml_path = '/data/DataSets/transmission_line_detection/train14000_xml'
classes = gen_classes(xml_path)
print(classes)

# 2.将类别名进行更改，因为有可能名字不是规范的，如果是规范的，就可以不用更改，不过建议可以过一下代码。

'''Modify the category name of the xml file to a specific name'''

def modify_one_xml(xml_path):
    file = open(xml_path)
    tree = ET.parse(file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls_name = obj.find('name')
        if cls_name.text == 'diaoche':
            cls_name.text = "DiaoChe"
        elif cls_name.text == 'shigongjixie':
            cls_name.text = "ShiGongJiXie"
        elif cls_name.text == 'daoxianyiwu':
            cls_name.text = "DaoXianYiWu"
        elif cls_name.text == 'tadiao':
            cls_name.text = "TaDiao"
        elif cls_name.text == 'yanhuo':
            cls_name.text = "YanHuo"
        elif cls_name.text == 'huoyan':
            cls_name.text = "YanHuo"
        else:
            pass
    tree.write(xml_path)

path = '/shared/xjd/DataSets/transmission_line_detection/train14000/Annotations'
filelist = os.listdir(path)
xml_name_list = [filename for filename in filelist if filename[-3:] == 'xml']
print(len(xml_name_list))
for xml_name in xml_name_list:
    modify_one_xml(os.path.join(path, xml_name))

print('Finished!')


# 3.给xml文件增加节点，因为我们自定义了数据格式，需要增加一个path节点，才能访问到图片。
import os
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

def add_xml_node(xml_file, tag, img_dir):
    '''
    往xml文件中增加一个节点
    Args:
        tag: 要增加的节点的名字，比如:'path','name'
        xml_file: 要增加节点的xml文件
    Returns:None
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    vars = root.findall(tag)
    if len(vars) != 0:    # 如果有path节点的话
        path = root.find(tag)
        # path.text = os.path.join(img_dir, os.path.basename(xml_file)[:-3] + 'jpg')
        path.text = os.path.basename(img_dir)
        print(path.text)
        tree.write(xml_file, encoding='utf-8')
        # print('走的if分支')
    else:
        element = Element(tag)
        # element.text = os.path.join(img_dir, os.path.basename(xml_file)[:-3] + 'jpg')
        element.text = os.path.basename(img_dir)
        # print(element.text)
        root.append(element)
        tree.write(xml_file, encoding='utf-8')
        # print('走的else分支')

    # print('Finished!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='add one node to a xml file!')
    parser.add_argument(
        "--xml-dir",
        default='/shared/xjd/DataSets/transmission_line_detection/test_demo_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--img-dir",
        default='/shared/xjd/DataSets/transmission_line_detection/test_demo',
        type=str,
        help='Directory path to jpg files.'
    )

    args = parser.parse_args()
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))    # 返回以.xml结尾的目录及文件列表
    print(f"Number of xml files:{len(xml_files)}")
    for xml_file in xml_files:
        add_xml_node(xml_file, 'path', args.img_dir)

    print('Finished!')


# 4.转为json。



