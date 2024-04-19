

# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml

device = select_device('0', batch_size=1)
# model trained by cfg/training/*.yaml
ckpt = torch.load('runs/train/yolov7-objects365/weights/best.pt', map_location=device)
# reparameterized model in cfg/deploy/*.yaml
model = Model('cfg/deploy/descriptor-yolov7.yaml', ch=3, nc=80).to(device)

with open('cfg/deploy/yolov7-objects365.yaml') as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
anchors = len(yml['anchors'][0]) // 2

# copy intersect weights
state_dict = ckpt['model'].float().state_dict()
exclude = []
intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
print(len(intersect_state_dict))
print(len(state_dict))
model.load_state_dict(intersect_state_dict, strict=False)
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

with torch.no_grad():
    ckpt['model'].model[-1].fuse()
model.model[-1].m = ckpt['model'].model[-1].m

# model to be saved
ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None,
        'training_results': None,
        'epoch': -1}

# save reparameterized model
torch.save(ckpt, 'runs/deploy/yolov7-objects365.pt')
torch.save(model.state_dict(), 'runs/deploy/yolov7-objects365-state_dict.pt')

