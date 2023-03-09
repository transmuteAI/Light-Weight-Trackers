class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/workspace/Stark'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/workspace/Stark/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/workspace/stark/pretrained_networks'
        self.lasot_dir = '/workspace/tracking_datasets/lasot/LaSOTBenchmark/'
        self.got10k_dir = '/workspace/tracking_datasets/got10k/train'
        self.lasot_lmdb_dir = '/workspace/tracking_datasets/lasot_lmdb'
        self.got10k_lmdb_dir = '/workspace/tracking_datasets/got10k_lmdb'
        self.trackingnet_dir = '/workspace/tracking_datasets/trackingnet'
        self.trackingnet_lmdb_dir = '/workspace/tracking_datasets/trackingnet_lmdb'
        self.coco_dir = '/workspace/tracking_datasets/coco'
        self.coco_lmdb_dir = '/workspace/tracking_datasets/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/workspace/tracking_datasets/vid'
        self.imagenet_lmdb_dir = '/workspace/tracking_datasets/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
