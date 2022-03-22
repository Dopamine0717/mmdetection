import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config
from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],    # 这三个pipeline排除,方便可视化
        help='skip some useless pipeline')
    # parser.add_argument(
    #     '--output-dir',
    #     default=None,
    #     type=str,
    #     help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=0,
        help='the interval of show (ms)')
    # 以下面这种方式指定可视化图像的保存路径,记得将最后的balloon换成我们自己数据集类型
    # TODO:有没有更好的方式，实现自动改变balloon，而不需要自己写？
    parser.add_argument(
        '--output_dir',
        default='/data/chenchao/MMDetection/customed_op_result/browse_dataset/transmission_line_detection/',
        type=str,
        help='Visual image save path')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    if train_data_cfg.get('dataset', None) is not None:
        # voc数据集
        datasets = train_data_cfg['dataset']
        datasets['pipeline'] = [
            x for x in datasets.pipeline if x['type'] not in skip_type
        ]
    else:
        # COCO数据集？对的
        train_data_cfg['pipeline'] = [
            x for x in train_data_cfg.pipeline if x['type'] not in skip_type
        ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name    # 带后缀的完整文件名
                                ) if args.output_dir is not None else None
        mmcv.imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'],
            class_names=dataset.CLASSES,
            show=not args.not_show,
            out_file=filename,
            wait_time=args.show_interval)
        progress_bar.update()


if __name__ == '__main__':
    main()
