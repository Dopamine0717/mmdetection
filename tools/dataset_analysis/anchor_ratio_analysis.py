import pandas as pd
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)

# load json file
ann_json = "/shared/xjd/DataSets/transmission_line_detection/train16462.json"
with open(ann_json, 'r') as f:
    ann = json.load(f)

category_dict = dict([(i['id'], i['name']) for i in ann['categories']])
counts_label = dict([(i['name'], 0) for i in ann['categories']])

for i in ann['annotations']:
    counts_label[category_dict[i['category_id']]] += 1

box_w = []
box_h = []
box_wh = []
categorys_wh = [[] for j in range(10)]
for a in ann['annotations']:
    # 不背景类不在category_id中
    box_w.append(round(a['bbox'][2],2))
    box_h.append(round(a['bbox'][3],2))
    wh = round(a['bbox'][2]/a['bbox'][3],0)
    if wh <1 :
        wh = round(a['bbox'][3]/a['bbox'][2],0)
    box_wh.append(wh)
    categorys_wh[a['category_id']].append(wh)

    # # 背景类在category_id中
    # if a['category_id'] != 0:
    #     box_w.append(round(a['bbox'][2],2))
    #     box_h.append(round(a['bbox'][3],2))
    #     wh = round(a['bbox'][2]/a['bbox'][3],0)
    #     if wh <1 :
    #         wh = round(a['bbox'][3]/a['bbox'][2],0)
    #     box_wh.append(wh)
    #     categorys_wh[a['category_id']-1].append(wh)

box_wh_unique = list(set(box_wh))
box_wh_count=[box_wh.count(i) for i in box_wh_unique]

wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['宽高比数量'])
wh_df.plot(kind='bar',color="#55aacc")
plt.xlabel('train set')
plt.savefig('anchor_ratio_train.png')
plt.show()











