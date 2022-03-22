import time
import numpy as np

def getAnchor(size, ratio, scale, feature_size, strides, names, root):
    num_base_anchors = len(ratio) * len(scale)
    result = []
    start = time.time()
    for index, s in enumerate(size):
        base_anchor = np.zeros((num_base_anchors, 4), dtype=np.float32)
        




