import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
# my import
from model import AIMnet
from dataset_all import TestData

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

bz = 1
model_root = 'pretrained/model.pth'
input_root = 'data/test/benchmarkA/'
save_path = 'result/benchmarkA'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=bz)

device_ids = list(range(torch.cuda.device_count()))
if not device_ids:
    raise RuntimeError("test.py requires at least one CUDA GPU.")

model = AIMnet().cuda()
model = nn.DataParallel(model, device_ids=device_ids)
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_dict'])
epoch = checkpoint['epoch']
model.eval()
print('START!')


def pad_to_multiple(tensor, multiple=4):
    _, _, height, width = tensor.size()
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect'), height, width


if 1:
    print('Load model successfully!')
    for data_idx, data_ in enumerate(data_load):
        data_input, data_la = data_

        data_input = Variable(data_input).cuda()
        data_la = Variable(data_la).cuda()
        print(data_idx)
        with torch.no_grad():
            data_input, origin_h, origin_w = pad_to_multiple(data_input)
            data_la, _, _ = pad_to_multiple(data_la)
            result, _ = model(data_input, data_la)
            result = result[:, :, :origin_h, :origin_w]
            name = os.path.basename(Mydata_.A_paths[data_idx])
            print(name)
            temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            temp_res = (temp_res*255).astype(np.uint8)
            temp_res = Image.fromarray(temp_res)
            temp_res.save('%s/%s' % (save_path, name))
            print('result saved!')

print('finished!')
