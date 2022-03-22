from tidecv import TIDE
import tidecv.datasets as datasets


# bbox_file = 

# gt = datasets.COCO()
# bbox_results = datasets.COCOResult(bbox_file)

# tide = TIDE()
# tide.evaluate_range(gt, bbox_results, mode=TIDE.BOX)
# tide.summarize()

import urllib.request # For downloading the sample Mask R-CNN annotations

bbox_file = 'mask_rcnn_bbox.json'
mask_file = 'mask_rcnn_mask.json'

urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/detectron/35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT/output/test/coco_2014_minival/generalized_rcnn/bbox_coco_2014_minival_results.json', bbox_file)
urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/detectron/35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT/output/test/coco_2014_minival/generalized_rcnn/segmentations_coco_2014_minival_results.json', mask_file)

print('Results Downloaded!')



