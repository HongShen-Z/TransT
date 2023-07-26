import torch
import pathlib
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone


path = '/home/test/zhs/projects/TransT/exps/checkpoints/normal/TransT_ep0120.pth.tar'

model = NetWithBackbone(net_path=path, use_gpu=True, initialize=True)
model = model.net

input_names = ['input']
output_names = ['output']

x = (torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 112, 112).cuda())

path = pathlib.Path(path)
to_path = pathlib.Path.joinpath(path.parent, path.name.split('.')[0] + '.onnx')

torch.onnx.export(model, x, to_path, input_names=input_names, output_names=output_names)
