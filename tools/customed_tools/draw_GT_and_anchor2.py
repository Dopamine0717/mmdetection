import argparse
import numpy as np
import cv2
import os
import os.path as osp
import torch
import mmcv
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmdet.core import build_anchor_generator

# TODO:需要实现对每试的一组参数，散点图都保存到相应的文件夹下，名字的话取名即为参数的组合
def parse_args():
    parser = argparse.ArgumentParser(description='Analyze GT and anchor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(  # dataloader parameter
        '--samples_per_gpu',
        type=int,
        default=16,
        help='batch size')
    parser.add_argument(  # dataloader parameter
        '--workers_per_gpu',
        type=int,
        default=8,
        help='worker num')
    parser.add_argument(
        '--out_path',
        type=str,
        default='train_7dirs_GT_data.npy',
        help='save GT data npy path')
    parser.add_argument(
        '--image-name',
        type=str,
        default='scatter_train_7dirs.png',
        help='save GT data npy path')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='/data/chenchao/personal_code/mmdetection/anchor_analyze/',
        help='save dir')
    parser.add_argument(  # When there is a local cache, whether to use it without going through the datalayer again
        '--use_local',
        type=bool,
        default=True,
        help='whether to use saved npy file')
    args = parser.parse_args()
    return args

def get_all_anchors(input_shape_hw, stride, anchor_generator_cfg):
    """Get all the anchors of a specific size image.
    
    Args:
        input_shape_hw: the height and width of a specific size image.
        stride: Stride of the feature map.
        anchor_generator_cfg: config of anchor generator.
    
    Return:
        all the anchors.
    """
    feature_map = []
    for s in stride:
        feature_map.append([input_shape_hw[0] // s, input_shape_hw[1] // s])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    anchors = anchor_generator.grid_priors(feature_map)
    all_anchors = []
    for anchor in anchors:    # len(anchors)=len(stride)
        anchor = anchor.cpu().numpy()

        # TODO:change to a more reasonable crop method.
        index = (anchor[:, 0] > 0) & (anchor[:, 1] > 0) & (anchor[:, 2] < input_shape_hw[1]) & \
                (anchor[:, 3] < input_shape_hw[0])
        anchor = anchor[index]
        anchor = np.random.permutation(anchor)
        all_anchors.append(anchor)
    return all_anchors

def get_all_GTs(cfg, args):
    """Get all the ground truths in the dataset.

    Args:
        cfg: Config of model.
        args: Parameters passed in from the command line.
    
    Return:
        all the ground truths.
    """
    use_local = args.use_local
    out_path = args.out_path
    if not use_local or not osp.isfile(out_path):
        print('--------重新获取数据---------')
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader(dataset, args.samples_per_gpu, args.workers_per_gpu)
        print('--------开始遍历数据集--------')
        all_GT = []
        progress_bar = mmcv.ProgressBar(len(dataloader))
        for i, data_batch in enumerate(dataloader):
            gt_bboxes = data_batch['gt_bboxes'].data[0]
            gt_bboxes = torch.cat(gt_bboxes, dim=0).numpy()
            if len(gt_bboxes) == 0:
                    continue
            xmin = gt_bboxes[:, 0]
            ymin = gt_bboxes[:, 1]
            xmax = gt_bboxes[:, 2]
            ymax = gt_bboxes[:, 3]
            GT = np.stack((xmin, ymin, xmax, ymax), axis=1)
            all_GT.append(GT)
            progress_bar.update()
        all_GT = np.concatenate(all_GT, axis=0)
        print(f"all ground truths' shape is {all_GT.shape}.")
        print('---------保存缓存文件--------')
        np.save(out_path, all_GT)
        print('--------文件保存完毕！-------')
    else:
        # Read data directly from the cache file.
        print('-------从缓存文件中读取-------')
        all_GT = np.load(out_path)
    return all_GT

def get_gt_wh(cfg, args):
    """Get all the ground truths in the dataset.
    Args:
        cfg: Config of model.
        args: Parameters passed in from the command line.
    
    Return:
        all the ground truths.
    """
    use_local = args.use_local
    out_path = args.out_path
    if not use_local or not osp.isfile(out_path):
        print('--------重新获取数据---------')
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader(dataset, args.samples_per_gpu, args.workers_per_gpu)
        print('--------开始遍历数据集--------')
        w_gt = []
        h_gt = []
        progress_bar = mmcv.ProgressBar(len(dataloader))
        for i, data_batch in enumerate(dataloader):
            gt_bboxes = data_batch['gt_bboxes'].data[0]
            gt_bboxes = torch.cat(gt_bboxes, dim=0).numpy()
            if len(gt_bboxes) == 0:
                    continue
            w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            w_gt.append(w)
            h_gt.append(h)
            progress_bar.update()
        w_gt = np.concatenate(w_gt, axis=0)
        h_gt = np.concatenate(h_gt, axis=0)
        print(f"all ground truths' shape is {w_gt.shape}.")
        # TODO:save data into a cache file
    return w_gt, h_gt

def plot_scatter(cfg, args):
    """plot scatter of the gt and anchor.
    Args:
        cfg: Config of model.
        args: Parameters passed in from the command line.
    """
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
    w_gt, h_gt = get_gt_wh(cfg, args)
    plt.scatter(w_anchor, h_anchor, color='hotpink', marker='x')
    plt.scatter(w_gt, h_gt, color='#88c999')
    plt.savefig('scatter.png')

def plot_scatter2(cfg, args):
    """plot scatter of the gt and anchor.
    Args:
        cfg: Config of model.
        args: Parameters passed in from the command line.
    """
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=3,
        scales_per_octave=3,
        ratios=[0.25, 0.6, 1.0, 1.8, 3.0],
        strides=[8, 16, 32, 64, 128])
    save_dir = f'{args.save_dir}' + f'obs_{anchor_generator_cfg["octave_base_scale"]}' \
        f'spo_{anchor_generator_cfg["scales_per_octave"]}' + f"ratios_{'_'.join([str(i) for i in anchor_generator_cfg['ratios']])}" \
            + f"strides_{'_'.join([str(i) for i in anchor_generator_cfg['strides']])}" \
                + f"{args.out_path.split('.')[0]}"
    # print(save_dir)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

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
    plt.scatter(w_gt, h_gt, color='#88c999', alpha=0.1)
    plt.scatter(w_anchor, h_anchor, color='hotpink', marker='x')

    plt.title("GT and anchors")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.legend(("gt", "anchors"), loc = 0)
    plt.savefig(f'{save_dir}/{args.image_name}')

def show_GT_and_anchors(cfg, args, input_shape_hw, stride, anchor_generator_cfg):
    """Visualize GT bbox and anchor bbox in object detection by drawing rectangle.
    
    Args:
        cfg: config of model.
        args: Parameters passed in from the command line.
        input_shape_hw: the height and width of a specific size image.
        stride: Stride of the feature map.
        anchor_generator_cfg: config of anchor generator.
    """
    img = np.zeros(input_shape_hw, np.uint8)
    all_GT = get_all_GTs(cfg, args)
    all_anchors = get_all_anchors(input_shape_hw, stride, anchor_generator_cfg)
    all_imgs = []
    for i, anchor in enumerate(all_anchors):
        img_ = show_bbox(img, anchor[:], color=(238,232,170), thickness=1, is_show=False, names=i)
        all_imgs.append(img_)
        show_bbox(img_, all_GT[:], color=(248,248,255), thickness=1, is_show=False, names=i)
    # TODO:func "show_img" add a parameter
    # show_img(all_imgs, stride, is_show=False)
    # img = np.zeros(input_shape_hw, np.uint8)
    # gt = show_bbox(img, all_GT[:], color=(248,248,255), thickness=1, is_show=False)


def merge_imgs(imgs, row_col_num):
    """Merges all input images as an image with specified merge format.
    
    Args:
        imgs: img list.
        row_col_num: number of rows and columns displayed.
    
    Return:
        img: merged img.
    """
    length = len(imgs)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
    
    # TODO:specify a color
    color = random_color(rgb=True).astype(np.float64)

    for img in imgs:
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

    if row_col_num[1] < 0 or length < row:
        merge_imgs = np.hstack(imgs)
    elif row_col_num[0] < 0 or length < col:
        merge_imgs = np.vstack(imgs)
    else:
        assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

        fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
        imgs.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(imgs[start: end])
            merge_imgs_col.append(merge_col)

        merge_imgs = np.vstack(merge_imgs_col)

    return merge_imgs

def show_img(imgs, window_names=None, is_show=False, wait_time_ms=0, is_merge=False, row_col_num=(1, -1)):
    """Displays an image or a list of images in specified windows or self-initiated windows.
    
    Args:
        imgs: numpy.ndarray or list.
        window_names: specified or None, if None, function will create different windows as '1', '2'.
        is_show (bool): whether to display during middle process. If false, imgs will be saved into files.
        wait_time_ms: display wait time. You can control display wait time by this parameter.
        is_merge: whether to merge all images. This parameter is to decide whether to display all 
            imgs in a particular window 'merge'.
        row_col_num: merge format. default is (1, -1), image will line up to show.
            example=(2, 5), images will display in two rows and five columns.
            Notice, specified format must be greater than or equal to imgs number.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    if window_names is None:
        window_names = list(range(len(imgs)))
    else:
        if not isinstance(window_names, list):
            window_names = [window_names]
        assert len(imgs) == len(window_names), 'window names does not match images!'

    if is_merge:
        merge_imgs1 = merge_imgs(imgs, row_col_num)

        cv2.namedWindow('merge', 0)
        cv2.imshow('merge', merge_imgs1)
    else:
        for img, win_name in zip(imgs, window_names):
            if img is None:
                continue
            win_name = str(win_name)
            if is_show:
                cv2.namedWindow(win_name, 0)
                cv2.imshow(win_name, img)
            else:
                # TODO:At present, imgs can only be kept in the project folder, find a way to save it in another folder.
                cv2.imwrite(f'{win_name}.png', img)
    if is_show:
        cv2.waitKey(wait_time_ms)
    else:
        pass

def show_bbox(image, bboxs_list, color=None,
              thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
              is_show=True, is_without_mask=False):
    """Visualize bbox in object detection by drawing rectangle.

    Args:
        image: numpy.ndarray.
        bboxs_list (list: [pts_xyxy, prob, id]): label or prediction.
        color: tuple.
        thickness: int.
        fontScale: float.
        wait_time_ms: int.
        names: string: window name.
        is_show (bool): whether to display during middle process.If false, imgs will be saved into files.
    
    Return: 
        numpy.ndarray.
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_copy = image.copy()
    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)
        if color is None:
            colors = random_color(rgb=True).astype(np.float64)
        else:
            colors = color

        if not is_without_mask:
            image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                       thickness)
        else:
            mask = np.zeros_like(image_copy, np.uint8)
            mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
            mask = np.zeros_like(image_copy, np.uint8)
            mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
            mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
            image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                        font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    if is_show:
        show_img(image_copy, names, wait_time_ms)
    else:
        # TODO:specify a name
        cv2.imwrite(f'GT&anchor{names}.jpg',image_copy)
    return image_copy

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret

def demo_retinanet(cfg, args, input_shape_hw):
    stride = [8, 16, 32, 64, 128]
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=2,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=stride)
    show_GT_and_anchors(cfg, args, input_shape_hw, stride, anchor_generator_cfg)

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # input_shape_hw = (1333, 1333, 3)
    # demo_retinanet(cfg, args, input_shape_hw)

    plot_scatter2(cfg, args)
    








