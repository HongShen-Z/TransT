class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/test/zhs/projects/TransT/checkpoints/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/home/test/zhs/datasets/SOT/LaSOT/'
        self.got10k_dir = '/home/test/zhs/datasets/GOT-10k/'
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
