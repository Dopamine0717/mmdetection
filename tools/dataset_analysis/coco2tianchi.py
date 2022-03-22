import json

with open("C:/Users/陈超/Desktop/instances_train.json", "r") as f:
    json_dict = json.load(f)

json_list = []
for annotation in json_dict["annotations"]:
    name = json_dict["images"][annotation["image_id"]]["file_name"]
    image_height = json_dict["images"][annotation["image_id"]]["height"]
    image_width = json_dict["images"][annotation["image_id"]]["width"]
    category = annotation["category_id"]
    bbox = annotation["bbox"]
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3] 
    annotation = {
            "name":name,
            "image_height":image_height,
            "image_width":image_width,
            "category":category,
            "bbox":bbox    # coco的bbox是[xmin, ymin, width, height],tianchi的要求是[xmin, ymin, xmax, ymax]
        }
    json_list.append(annotation)


json_file = "C:/Users/陈超/Desktop/coco2tianchi_train.json"
json_fp = open(json_file, 'w')
json_str = json.dumps(json_list, ensure_ascii=False)
json_fp.write(json_str)
json_fp.close()