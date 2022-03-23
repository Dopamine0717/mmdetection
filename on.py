import torch
from distutils.command.build import build
from mmdet.core.export import preprocess_example_input
from mmdet.core.export import build_model_from_cfg, preprocess_example_input
model = build_model_from_cfg(config_path="./configs/transmission_line_detection202/retinanet_r50_fpn_1x_transmission_custom_lr0.02.py",
                             checkpoint_path="checkpoints/epoch_50.pth")


input_config = {
'input_shape': (1,3,640,640),
'input_path': '3_640_640.jpg',
'normalize_cfg': {
'mean': (123.675, 116.28, 103.53),
 'std': (58.395, 57.12, 57.375)
}
}

one_img, one_meta = preprocess_example_input(input_config)
print(one_img.shape)
print(one_meta)

model.forward = model.forward_dummy
torch.onnx.export(
            model,
            one_img,
            output_file='tmp.onnx',
            input_names=['input'],
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)


