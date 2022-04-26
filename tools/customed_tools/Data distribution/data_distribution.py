import os
import random
import shutil


filelist = os.listdir('F:/数据集/数据分配/images')
xiaohan = filelist[:500]    # 这里之所以都是0：500，是因为是将文件进行一动而不是将文件复制
yukun = filelist[:500]
haichen = filelist[:500]
mingxuan = filelist[:500]
zhaoshuang = filelist[:500]

images_path = r'F:\数据集\数据分配\images'
xml_path = r'F:\数据集\数据分配\xml'

# for filename in xiaohan:
#     name = filename[:-4]
#     path = 'F:/数据集/数据分配/晓涵'
#     jpg_src = name + '.jpg'
#     xml_src = name + '.xml'
#
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     shutil.move(os.path.join(images_path, jpg_src), path)
#     shutil.move(os.path.join(xml_path, xml_src), path)
# print('Finished!')

# for filename in yukun:
#     name = filename[:-4]
#     path = 'F:/数据集/数据分配/玉坤'
#     jpg_src = name + '.jpg'
#     xml_src = name + '.xml'
#
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     shutil.move(os.path.join(images_path, jpg_src), path)
#     shutil.move(os.path.join(xml_path, xml_src), path)
# print('Finished!')

# for filename in haichen:
#     name = filename[:-4]
#     path = 'F:/数据集/数据分配/海宸'
#     jpg_src = name + '.jpg'
#     xml_src = name + '.xml'
#
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     shutil.move(os.path.join(images_path, jpg_src), path)
#     shutil.move(os.path.join(xml_path, xml_src), path)
# print('Finished!')

# for filename in mingxuan:
#     name = filename[:-4]
#     path = 'F:/数据集/数据分配/明轩'
#     jpg_src = name + '.jpg'
#     xml_src = name + '.xml'
#
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     shutil.move(os.path.join(images_path, jpg_src), path)
#     shutil.move(os.path.join(xml_path, xml_src), path)
# print('Finished!')

for filename in zhaoshuang:
    name = filename[:-4]
    path = 'F:/数据集/数据分配/赵爽'
    jpg_src = name + '.jpg'
    xml_src = name + '.xml'

    if not os.path.exists(path):
        os.makedirs(path)

    shutil.move(os.path.join(images_path, jpg_src), path)
    shutil.move(os.path.join(xml_path, xml_src), path)
print('Finished!')