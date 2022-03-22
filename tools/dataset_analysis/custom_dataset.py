import os

xml_path = '/shared/xjd/chenchao/dataset/coco_custom/transmission_line_detection/test_xml'
image_path = '/shared/xjd/chenchao/dataset/coco_custom/transmission_line_detection/test'

# xml_path = r'F:\数据集\voc2coco-master\test\Annotations'
# image_path = r'F:\数据集\voc2coco-master\test\images'
# txt_path = r'F:\数据集\voc2coco-master\test\labels'
def renumber(xml_path, img_path, txt_path=None):
    xml_list = os.listdir(xml_path)
    img_list = os.listdir(img_path)
    print(len(xml_list))
    print(len(img_list))

    i = 13172    # 起始编号
    xml_total_num = len(xml_list)
    img_total_num = len(img_list)

    for item in xml_list:
        if item.endswith('.xml'):
            src1 = os.path.join(os.path.abspath(xml_path),item)
            dst1 = os.path.join(os.path.abspath(xml_path),str(i) + '.xml')
            src2 = os.path.join(os.path.abspath(img_path), item[:-3] +'jpg')
            dst2 = os.path.join(os.path.abspath(img_path), str(i) + '.jpg')

            # # TODO:如果不需要改名txt，则注释掉下面的
            # src3 = os.path.join(os.path.abspath(txt_path), item[:-3] + 'txt')
            # dst3 = os.path.join(os.path.abspath(txt_path), str(i) + '.txt')

            os.rename(src1, dst1)
            print('Converting %s to %s ...' % (src1, dst1))
            os.rename(src2, dst2)
            print('Converting %s to %s ...' % (src2, dst2))

            # # TODO:如果不需要改名txt，则注释掉下面的
            # os.rename(src3, dst3)
            # print('Converting %s to %s ...' % (src3, dst3))

            i += 1

    print('Total %d to rename & converted %d xmls' % (xml_total_num, i - 1))
    print('Total %d to rename & converted %d jpgs' % (img_total_num, i - 1))

if __name__ == '__main__':
    # renumber(xml_path, image_path, txt_path)
    renumber(xml_path, image_path)
