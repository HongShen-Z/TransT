from TrackingNet.Downloader import TrackingNetDownloader
from TrackingNet.utils import getListSplit


downloader = TrackingNetDownloader(LocalDirectory="/home/test/zhs/datasets/TrackingNet")

for split in getListSplit()[1:5]:
    # print(split)
    downloader.downloadSplit(split)
