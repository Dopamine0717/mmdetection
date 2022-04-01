# 将标注的数据转换成json格式
# ====================================分隔线===============================
# 1.将所有的文件夹进行归纳汇总,移动到相应的文件夹
import os
import shutil

img_path = [
    'D:/数据/第二批杆塔/陈玉坤/cyk',
    'D:/数据/第二批杆塔/李明轩/lmx',
    'D:/数据/第二批杆塔/罗海宸/lhc',
    'D:/数据/第二批杆塔/赵爽/zs',
    'D:/数据/第二批杆塔/赵晓涵/zxh'
]
xml_path = [
    'D:/数据/第二批杆塔/陈玉坤/cyk_xml',
    'D:/数据/第二批杆塔/李明轩/lmx_xml',
    'D:/数据/第二批杆塔/罗海宸/lhc_xml',
    'D:/数据/第二批杆塔/赵爽/zs_xml',
    'D:/数据/第二批杆塔/赵晓涵/zxh_xml'
]
img_save_path = 'D:/数据/第二批杆塔/data1_add_ganta1'    # 图片保存路径
xml_save_path = 'D:/数据/第二批杆塔/data1_add_ganta1_xml'    # xml文件保存路径

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)
if not os.path.exists(xml_save_path):
    os.makedirs(xml_save_path)
for i in range(len(img_path)):
    filelist = os.listdir(img_path[i])
    for filename in filelist:
        if filename[-4:] == '.jpg':
            shutil.move(os.path.join(img_path[i], filename), img_save_path)

for i in range(len(xml_path)):
    filelist = os.listdir(xml_path[i])
    for filename in filelist:
        if filename[-4:] == '.xml':
            shutil.move(os.path.join(xml_path[i], filename), xml_save_path)  

print('Finished!')

# ====================================分隔线===============================
# 2.查看类别数量以及文件数量
import xml.etree.cElementTree as ET
import os

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

# xml_path = '/shared/xjd/DataSets/transmission_line_detection/train14000_xml'
xml_path = xml_save_path
classes = gen_classes(xml_path)
print(classes)

# ====================================分隔线===============================
import os

def check_file_nums(dir_path, recursion=False):
    '''This function can help us check the file numbers of the dir_path.
    
    Args:
        dir_path (str): The path of the folder to check.
        recursion (bool): whether to preform recursive query. Default is False.

    Return: 
        numbers of file, numbers of dir
    '''
    if os.path.isdir(dir_path):
        if not recursion:
            filelist = [i for i in os.listdir(dir_path) if not os.path.isdir(os.path.join(dir_path,i))]
            return len(filelist), 0
        else:
            filelist = os.listdir(dir_path)
            file_nums = 0
            dir_nums = 0
            for i in filelist:
                if os.path.isdir(os.path.join(dir_path,i)):
                    dir_nums += 1
                    file_num, dir_num = check_file_nums(os.path.join(dir_path,i), recursion)
                    file_nums += file_num
                    dir_nums += dir_num
                else:
                    file_nums += 1
            return file_nums, dir_nums
    else:
        print("The path you entered is not a folder!")

file_nums, dir_nums = check_file_nums(img_save_path, recursion=True)
print(f"The number of img file is {file_nums}")
print(f"The number of dir is {dir_nums}")
file_nums, dir_nums = check_file_nums(xml_save_path, recursion=True)
print(f"The number of xml file is {file_nums}")
print(f"The number of dir is {dir_nums}")

# ====================================分隔线===============================
# 3.将类别名进行更改  在标注没有错误的情况下，就是走个形式，有错误的情况下可以进行矫正
import xml.etree.cElementTree as ET
import os

def modify_one_xml(xml_path):
    file = open(xml_path)
    tree = ET.parse(file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls_name = obj.find('name')
        if cls_name.text == 'diaoche':
            cls_name.text = "DiaoChe"
            print('DiaoChe 进行了修改')
        elif cls_name.text == 'shigongjixie':
            cls_name.text = "ShiGongJiXie"
            print('ShiGongJiXie 进行了修改')
        elif cls_name.text == 'daoxianyiwu':
            cls_name.text = "DaoXianYiWu"
            print('DaoXianYiWu 进行了修改')
        elif cls_name.text == 'tadiao':
            cls_name.text = "TaDiao"
            print('TaDiao 进行了修改')
        elif cls_name.text == 'yanhuo':
            cls_name.text = "YanHuo"
            print('YanHuo 进行了修改')
        elif cls_name.text == 'huoyan':
            cls_name.text = "YanHuo"
            print('YanHuo 进行了修改')
        elif cls_name.text == 'ganta':
            cls_name.text = "GanTa"
            print('GanTa 进行了修改')
        else:
            pass
    tree.write(xml_path)

# path = '/shared/xjd/DataSets/transmission_line_detection/train14000_xml'
path = xml_save_path
filelist = os.listdir(path)
xml_name_list = [filename for filename in filelist if filename[-3:] == 'xml']
print(len(xml_name_list))
for xml_name in xml_name_list:
    modify_one_xml(os.path.join(path, xml_name))

print('Finished!')

# ====================================分隔线===============================
# 4.增加path节点或者修改path节点
import os
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

def add_xml_node(xml_file, tag, img_dir):    # 这里的img_dir实际上是可以自己定义的，如果在自己电脑上转的话比较有用
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
        path.text = os.path.basename(img_dir)
        print(path.text)
        tree.write(xml_file, encoding='utf-8')
        # print('走的if分支')
    else:
        element = Element(tag)
        element.text = os.path.basename(img_dir)
        root.append(element)
        tree.write(xml_file, encoding='utf-8')
        # print('走的else分支')

xml_dir = xml_save_path
img_dir = '/shared/xjd/DataSets/transmission_line_detection/train'
    
xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))    # 返回以.xml结尾的目录及文件列表
print(f"Number of xml files:{len(xml_files)}")
for xml_file in xml_files:
    add_xml_node(xml_file, 'path', img_dir)

print('Finished!')

# ====================================分隔线===============================
# 5.xml转为json
import glob
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import json


START_BOUNDING_BOX_ID = 0
START_IMAGE_ID = 0
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

        

        # TODO:在不将文件名更改的情况下，实现转换。主要依据其实就是一个xml文件对应一张image，除后缀外，名字相同，因此有了下面的代码。
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


xml_dir = 'D:/数据/第二批杆塔/data1_add_ganta1_xml'
json_file = 'D:/数据/第二批杆塔/data1_add_ganta1.json'
xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))    # 返回以.xml结尾的目录及文件列表

print(f"Number of xml files:{len(xml_files)}")
convert(xml_files, json_file)
print(f"Success:{json_file}")

