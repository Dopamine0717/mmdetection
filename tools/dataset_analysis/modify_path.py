import json

with open("work_dirs_semi_supervision/train14000/train14000.json.bbox.json", "r") as f:
    json_dict = json.load(f)

for img_info in json_dict["images"]:
    if img_info["path"] != None:
        img_info["path"] = "train14000"
json_file = "work_dirs_semi_supervision/train14000/train14000.json"
json_fp = open(json_file, 'w')
json_str = json.dumps(json_dict, ensure_ascii=False)
json_fp.write(json_str)
json_fp.close()