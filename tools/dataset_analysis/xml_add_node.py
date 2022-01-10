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




