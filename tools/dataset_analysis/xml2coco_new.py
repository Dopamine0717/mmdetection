import glob
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import json
import random


START_BOUNDING_BOX_ID = 0
START_IMAGE_ID = 0
# PRE_DEFINE_CATEGORIES = {"DaoXianYiWu": 0, "DiaoChe": 1, "ShiGongJiXie": 2, "TaDiao": 3, "YanHuo":4}
# PRE_DEFINE_CATEGORIES = {"GanTa": 0}
PRE_DEFINE_CATEGORIES = {"DaoXianYiWu": 0, "DiaoChe": 1, "ShiGongJiXie": 2, "TaDiao": 3, "YanHuo":4, "GanTa":5}

def get_categories(xml_files):
    '''
    Generte category name to id mapping from a list of xml files.
    Args:
        xml_files[list]: A list of xml file paths.
    Return: dict -- category name to id mapping.
    '''
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for item in root.findall("object"):
            classes_names.append(item.find('name').text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name : i for i, name in enumerate(classes_names)}

def get_and_check(root, tag, length):
    '''
    Args:
        root: xml.etree.ElementTree.ElementTree object
        tag: xml file tag name. eg:"size","width"
        length: default 1
    Return: filename
    '''
    vars = root.findall(tag)
    if len(vars) == 0:
        raise ValueError(f"Can not find {tag} in {root.tag}")
    if length > 0 and len(vars) != length:
        raise ValueError(
            f"The size of {tag} is supposed to be {length}, but is {len(vars)}."
        )
    if length == 1:
        vars = vars[0]
    return vars

def get(root, tag):
    vars = root.findall(tag)
    return vars

# 这个函数可以单独拎出来
def add_xml_node(xml_file, tag, img_dir):
    '''
    往xml文件中增加一个节点
    Args:
        tag: 要增加的节点的名字，比如：'path','name'
        xml_file: 要增加节点的xml文件
    Returns:None
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    vars = root.findall(tag)
    if len(vars) != 0:    # 如果有path节点的话
        path = root.find(tag)
        path.text = os.path.join(img_dir, os.path.basename(xml_file)[:-3] + 'jpg')
        print(path.text)
        tree.write(xml_file, encoding='utf-8')
        # print('走的if分支')
    else:
        element = Element(tag)
        element.text = os.path.join(img_dir, os.path.basename(xml_file)[:-3] + 'jpg')
        # print(element.text)
        root.append(element)
        tree.write(xml_file, encoding='utf-8')
        # print('走的else分支')
    # print('Finished!')

def convert(xml_files, json_file):
    '''
    Convert xml annotations to COCO format.
    Args:
        xml_file: xml format file path.
        json_file: output to a json file.
    Return: None
    '''
    json_dict = {
        "images":[],
        "type":"instances",
        "annotations":[],
        "categories":[]
    }
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)

    image_id = START_IMAGE_ID
    bbox_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get_and_check(root, "path", 1).text

        # 这里假设我们的文件是以数字命名的，比如："1.jpg","2.jpg","1.xml","2.xml"
        # filename = os.path.basename(xml_file)[:-3] + 'jpg'
        # image_id = int(os.path.basename(xml_file)[:-4])

        # TODO:在不将文件名更改的情况下，实现转换。主要依据其实就是一个xml文件对应一张image，除后缀外，名字相同，因此有了下面的代码。
        # filename = os.path.basename(path)    # 这是path是完整的路径时的用法
        filename = os.path.basename(xml_file)[:-3] + 'jpg'    # 这是path只是一个文件夹名字的用法

        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {
            "path":path,
            "file_name":filename,
            "height":height,
            "width":width,
            "id":image_id
        }
        json_dict["images"].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:    # 事实上，如果只想对已经指定的某些类别进行转化，这里只需pass就好，不需要创建新的类别映射
                # new_id = len(categories)
                # categories[category] = new_id
                continue
            
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)

            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert xmax > xmin, f'{xml_file}'
            assert ymax > ymin, f'{xml_file}'
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area":o_width * o_height,
                "iscrowd":0,
                "image_id":image_id,
                "bbox":[xmin, ymin, o_width, o_height],
                "category_id":category_id,
                "id":bbox_id,    # 这个表示object的id
                "ignore":0,
                "segmentation":[]
            }
            json_dict["annotations"].append(ann)
            bbox_id += 1
        image_id += 1
    for cate, cid in categories.items():
        cat = {
            "supercategory":"none",
            "id":cid,
            "name":cate
        }
        json_dict['categories'].append(cat)
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, ensure_ascii=False)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert xml annotations to COCO format!')
    parser.add_argument(
        "--xml-dir", 
        default='/shared/xjd/DataSets/transmission_line_detection/train_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--xml-dir2", 
        default='/shared/xjd/DataSets/transmission_line_detection/daoxianyiwu_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--xml-dir3", 
        default='/shared/xjd/DataSets/transmission_line_detection/fire_detection_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--xml-dir3", 
        default='/shared/xjd/DataSets/transmission_line_detection/test_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--xml-dir4", 
        default='/shared/xjd/DataSets/transmission_line_detection/train14000_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--xml-dir5", 
        default='/shared/xjd/DataSets/transmission_line_detection/test_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--xml-dir6", 
        default='/shared/xjd/DataSets/transmission_line_detection/train14000_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--xml-dir7", 
        default='/shared/xjd/DataSets/transmission_line_detection/train14000_xml',
        type=str,
        help='Directory path to xml files.'
        )
    parser.add_argument(
        "--json-file",
        default='/shared/xjd/DataSets/transmission_line_detection/data1_add_dxyw.json',
        type=str,
        help='Output COCO format json file.'
        )
    # parser.add_argument(
    #     "--json-file2",
    #     default='/shared/xjd/DataSets/transmission_line_detection/test_6cates_3490.json',
    #     type=str,
    #     help='Output COCO format json file.'
    #     )
    args = parser.parse_args()
    # ----------------------------------------------------------------
    # xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))    # 返回以.xml结尾的目录及文件列表
    
    # # 下面的代码只有在结合两个来源的数据的时候用到
    # xml_files2 = glob.glob(os.path.join(args.xml_dir2, "*.xml"))
    # xml_files.extend(xml_files2[2500:])
    # xml_files3 = glob.glob(os.path.join(args.xml_dir3, "*.xml"))
    # xml_files.extend(xml_files3)

    # # 大数据集时才需要下面的代码
    # xml_files4 = glob.glob(os.path.join(args.xml_dir4, "*.xml"))
    # xml_files.extend(xml_files4[:-200])
    # xml_files5 = glob.glob(os.path.join(args.xml_dir5, "*.xml"))
    # xml_files.extend(xml_files5)
    # xml_files6 = glob.glob(os.path.join(args.xml_dir6, "*.xml"))
    # xml_files.extend(xml_files6)

    # print(f"Number of xml files:{len(xml_files)}")
    # convert(xml_files, args.json_file)
    # print(f"Success:{args.json_file}")

    # print(f"Number of xml files:{len(xml_files3)}")
    # convert(xml_files3, args.json_file2)
    # print(f"Success:{args.json_file2}")
    # -------------------------------------------------------------
    
    
    # -------------------------------------------------------------
    # train_files = []
    # test_files = []
    # xml_files = random.shuffle(glob.glob(os.path.join(args.xml_dir, "*.xml")))    # 返回以.xml结尾的目录及文件列表
    # xml_files2 = random.shuffle(glob.glob(os.path.join(args.xml_dir2, "*.xml")))
    # xml_files2 = xml_files2[2500:]
    # xml_files3 = random.shuffle(glob.glob(os.path.join(args.xml_dir3, "*.xml")))
    # xml_files4 = random.shuffle(glob.glob(os.path.join(args.xml_dir4, "*.xml")))
    # xml_files5 = random.shuffle(glob.glob(os.path.join(args.xml_dir5, "*.xml")))
    # xml_files6 = random.shuffle(glob.glob(os.path.join(args.xml_dir6, "*.xml")))
    # xml_files7 = random.shuffle(glob.glob(os.path.join(args.xml_dir7, "*.xml")))
    
    # train_files.extend(xml_files[:554])
    # train_files.extend(xml_files2[:554])
    # train_files.extend(xml_files3[:554])
    # train_files.extend(xml_files4[:554])
    # train_files.extend(xml_files5[:554])
    # train_files.extend(xml_files6[:554])
    # train_files.extend(xml_files7[:554])
    
    # test_files.extend(xml_files[:554])
    # test_files.extend(xml_files2[:554])
    # test_files.extend(xml_files3[:554])
    # test_files.extend(xml_files4[:554])
    # test_files.extend(xml_files5[:554])
    # test_files.extend(xml_files6[:554])
    # test_files.extend(xml_files7[:554])

    # print(f"Number of xml files:{len(xml_files)}")
    # convert(xml_files, args.json_file)
    # print(f"Success:{args.json_file}")
    # -------------------------------------------------------------
    
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))    # 返回以.xml结尾的目录及文件列表
    random.shuffle(xml_files)
    
    xml_files2 = glob.glob(os.path.join(args.xml_dir2, "*.xml"))
    xml_files2 = xml_files2[2500:]
    random.shuffle(xml_files2)
    xml_files2_DiaoChe = [i for i in xml_files2 if os.path.basename(i)[:3] == 'Dia']
    xml_files2_SGJX = [i for i in xml_files2 if os.path.basename(i)[:3] == 'SGJ']
    xml_files2_TaDiao = [i for i in xml_files2 if os.path.basename(i)[:3] == 'TaD']
    xml_files2_DXYW = [i for i in xml_files2 if os.path.basename(i)[:3] == 'DXY']
    xml_files2_YanHuo = [i for i in xml_files2 if os.path.basename(i)[:3] == 'Yan']
    print(len(xml_files2_DiaoChe), len(xml_files2_SGJX), len(xml_files2_TaDiao), len(xml_files2_DXYW), len(xml_files2_YanHuo))
    
    
    xml_files3 = glob.glob(os.path.join(args.xml_dir3, "*.xml"))
    random.shuffle(xml_files3)
    
    xml_files4 = glob.glob(os.path.join(args.xml_dir4, "*.xml"))
    random.shuffle(xml_files4)
    
    xml_files5 = glob.glob(os.path.join(args.xml_dir5, "*.xml"))
    random.shuffle(xml_files5)
    xml_files5_DiaoChe = [i for i in xml_files5 if os.path.basename(i)[:3] == 'Dia']
    xml_files5_SGJX = [i for i in xml_files5 if os.path.basename(i)[:3] == 'SGJ']
    xml_files5_TaDiao = [i for i in xml_files5 if os.path.basename(i)[:3] == 'TaD']
    xml_files5_DXYW = [i for i in xml_files5 if os.path.basename(i)[:3] == 'DXY']
    xml_files5_YanHuo = [i for i in xml_files5 if os.path.basename(i)[:3] == 'Yan']
    print(len(xml_files5_DiaoChe), len(xml_files5_SGJX), len(xml_files5_TaDiao), len(xml_files5_DXYW), len(xml_files5_YanHuo))
    
    xml_files6 = glob.glob(os.path.join(args.xml_dir6, "*.xml"))
    random.shuffle(xml_files6)
    
    xml_files7 = glob.glob(os.path.join(args.xml_dir7, "*.xml"))
    random.shuffle(xml_files7)
    xml_files7_DiaoChe = [i for i in xml_files7 if os.path.basename(i)[:3] == 'Dia']
    xml_files7_SGJX = [i for i in xml_files7 if os.path.basename(i)[:3] == 'SGJ']
    xml_files7_TaDiao = [i for i in xml_files7 if os.path.basename(i)[:3] == 'TaD']
    xml_files7_DXYW = [i for i in xml_files7 if os.path.basename(i)[:3] == 'DXY']
    xml_files7_YanHuo = [i for i in xml_files7 if os.path.basename(i)[:3] == 'Yan']
    print(len(xml_files7_DiaoChe), len(xml_files7_SGJX), len(xml_files7_TaDiao), len(xml_files7_DXYW), len(xml_files7_YanHuo))
    
    train_files = []
    test_files = []
    
    train_files.extend(xml_files[:530])
    train_files.extend(xml_files2)
    train_files.extend(xml_files3[:170])
    train_files.extend(xml_files4)
    train_files.extend(xml_files5[:2100])
    train_files.extend(xml_files6[:1710])
    train_files.extend(xml_files7_DiaoChe[:668])
    train_files.extend(xml_files7_SGJX[:250])
    train_files.extend(xml_files7_TaDiao[:220])
    train_files.extend(xml_files7_DXYW[:12])
    train_files.extend(xml_files7_YanHuo[:44])
    
    test_files.extend(xml_files[530:])
    test_files.extend(xml_files3[170:])
    test_files.extend(xml_files5[2100:])
    test_files.extend(xml_files6[1710:])
    test_files.extend(xml_files7_DiaoChe[668:])
    test_files.extend(xml_files7_SGJX[250:])
    test_files.extend(xml_files7_TaDiao[220:])
    test_files.extend(xml_files7_DXYW[12:])
    test_files.extend(xml_files7_YanHuo[44:])
    
    print(len(train_files), len(test_files))

    print(f"Number of xml files:{len(train_files)}")
    convert(train_files, args.json_file)
    print(f"Success:{args.json_file}")
    
    print(f"Number of xml files:{len(test_files)}")
    convert(test_files, args.json_file2)
    print(f"Success:{args.json_file2}")