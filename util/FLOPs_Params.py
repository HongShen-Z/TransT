from thop import profile
from thop.utils import clever_format
import torch
import ltr.models.tracking.transt as transt_models
# from efficientnet_pytorch.utils import Conv2dDynamicSamePadding
# from efficientnet_pytorch.utils import Conv2dStaticSamePadding
# from efficientnet_pytorch.utils import MemoryEfficientSwish
# from thop.vision.basic_hooks import count_convNd, zero_ops

class Settings():
    def __init__(self):
        self.device = 'cuda'

if __name__ == "__main__":
    # Compute the Flops and Params
    # create model
    settings = Settings()
    model = transt_models.transt_multimax(settings)
    # print(model)
    backbone = model.backbone
    head = model.head

    x = torch.randn(1, 3, 256, 256).to(settings.device)
    zf = torch.randn(1, 3, 112, 112).to(settings.device)            # (1, 96, 8, 8)

    inp = torch.randn(1, 64, 16, 16).to(settings.device)
    # oup = model(x, zf)
    # print(oup['pred_logits'].shape, oup['pred_boxes'].shape)

    # custom_ops = {
    #     Conv2dDynamicSamePadding: count_convNd,
    #     Conv2dStaticSamePadding: count_convNd,
    #     MemoryEfficientSwish: zero_ops,
    # }
    # compute FLOPs and Params
    # the whole model
    macs, params = profile(model, inputs=(x, zf), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
    # backbone
    macs, params = profile(backbone, inputs=(x,), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('backbone macs is ', macs)
    print('backbone params is ', params)
    # head
    macs, params = profile(head, inputs=(inp,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('head macs is ', macs)
    print('head params is ', params)
