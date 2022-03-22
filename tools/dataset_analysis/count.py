import numpy
import json
with open("annotations/instances_val2017.json", 'r') as f:
    result = json.load(f)  # 读取json文件
    area=numpy.zeros(3)   
    proportion=numpy.zeros(3)
    for i in result["annotations"]:  # 读取annotations字段中的每个字段
        if i['area']<=32**2:
            area[0]+=1
        elif i['area']>32**2 and i['area']<96**2:
            area[1]+=1
        elif i['area']>=96**2:
            area[2]+=1
    for i in range(0,3):
        proportion[i]=area[i]/(area.sum())  # 占比
    print('小物体有{}个，占比{}%'.format(area[0],proportion[0].round(3)*100))
    print('中物体有{}个，占比{}%'.format(area[1],proportion[1].round(3)*100))
    print('大物体有{}个，占比{}%'.format(area[2],proportion[2].round(3)*100))