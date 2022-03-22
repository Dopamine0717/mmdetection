import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs
import albumentations as albu



# 1.训练集和测试集的数量
# Setup the paths to train and test images
TRAIN_DIR = '/shared/xjd/DataSets/transmission_line_detection/train/'
TEST_DIR = '/shared/xjd/DataSets/transmission_line_detection/test/'
TRAIN_CSV_PATH = '/shared/xjd/DataSets/transmission_line_detection/coco2tianchi_train.json'

# Glob the directories and get the lists of train and test images
train_fns = glob(TRAIN_DIR + '*')
test_fns = glob(TEST_DIR + '*')
print('Number of train images is {}'.format(len(train_fns)))
print('Number of test images is {}'.format(len(test_fns)))

# 2.每张图bbox的数量
import json
train = []
with open(TRAIN_CSV_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
train = pd.DataFrame(data)

train.head(5)

all_train_images = pd.DataFrame([fns.split('\\')[-1] for fns in train_fns])
all_train_images.columns=['name']
# merge image with json info
all_train_images = all_train_images.merge(train, on='name', how='left')
# replace nan values with zeros
all_train_images['bbox'] = all_train_images.bbox.fillna('[0,0,0,0]')
all_train_images.head(5)














