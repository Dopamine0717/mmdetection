import argparse
import numpy as np
import matplotlib.pyplot as plt
from mmdet.core import build_anchor_generator
from mmcv import Config
from draw_GT_and_anchor import get_all_GTs

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze GT and anchor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--out_path',
        type=str,
        default='GT_data.npy',
        help='save GT data npy path')
    parser.add_argument(  # When there is a local cache, whether to use it without going through the datalayer again, saving time
        '--use_local',
        type=bool,
        default=True,
        help='whether to use saved npy file')
    args = parser.parse_args()
    return args

def plot_scatter2(cfg, args):
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    w_anchor = []
    h_anchor = []
    for i in range(len(anchor_generator.base_anchors)):
        base_anchors = anchor_generator.base_anchors[i]
        base_anchors = base_anchors[:,:].cpu().numpy()
        w = base_anchors[:,2] - base_anchors[:,0]
        h = base_anchors[:,3] - base_anchors[:,1]
        w_anchor.append(w)
        h_anchor.append(h)
    w_anchor = np.concatenate(w_anchor, axis=0)
    h_anchor = np.concatenate(h_anchor, axis=0)
    all_GT = get_all_GTs(cfg, args)
    w_gt = all_GT[:,2] - all_GT[:,0]
    h_gt = all_GT[:,3] - all_GT[:,1]
    plt.scatter(w_gt, h_gt, color='#88c999')
    plt.scatter(w_anchor, h_anchor, color='hotpink', marker='x')

    plt.title("GT and anchors")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.legend(("gt", "anchors"), loc = 0)
    plt.savefig('scatter.png')

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    plot_scatter2(cfg, args)