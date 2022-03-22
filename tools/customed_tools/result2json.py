import mmcv

# TODO:将检测的结果转化成json格式，以实现半监督任务
def result2json(results, dataset, jsonfile_prefix):
    """Convert the detection results to json annotation format.

    Args:
        results(list[list | tuple | ndarray]): Testing results of the
            dataset.
        dataset: uesd to get the image information.
        outfile_prefix(str): The filename prefix of the json files.
    """
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = f'{jsonfile_prefix}.bbox.json'
        result_files['proposal'] = f'{jsonfile_prefix}.bbox.json'
        mmcv.dump(json_results, result_files['bbox'])
    else:
        raise TypeError('invalid type of results')
    return result_files

def det2json(dataset, results):
    """Convert detection results to COCO json style."""
    json_dict = {
        "images":[],
        "type":"instances",
        "annotations":[],
        "categories":[]
    }

    # 处理类别的逻辑
    for cid, cate in enumerate(dataset.CLASSES):
        cat = {
            "supercategory":"none",
            "id":cid,    # TODO:确认一下cid和cate没有搞反
            "name":cate
        }
    json_dict['categories'].append(cat)

    bbox_id = 0
    for idx in range(len(dataset)):
        image_id = dataset.img_ids[idx]
        path = dataset.ann_file    # TODO:要除去前面的一部分路径
        filename = dataset.data_infos[idx]["file_name"]
        height = dataset.data_infos[idx]["height"]
        width = dataset.data_infos[idx]["width"]
        image = {
            "path":path,
            "file_name":filename,
            "height":height,
            "width":width,
            "id":image_id
        }
        json_dict["images"].append(image)

        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                bbox = dataset.xyxy2xywh(bboxes[i])    # TODO:查看这里是不是一个list？以及注意xy是否对应左上角的坐标？
                xmin = bbox[0]
                ymin = bbox[1]
                o_width = bbox[2]
                o_height = bbox[3]
                category_id = dataset.cat_ids[label]
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
                bbox_id = bbox_id + 1

    return json_dict






