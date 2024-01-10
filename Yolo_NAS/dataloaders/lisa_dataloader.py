from super_gradients.common.registry.registry import register_dataloader
from super_gradients.common.object_names import Dataloaders
from super_gradients.training.dataloaders import get_data_loader
from torch.utils.data import DataLoader
# from super_gradients.training.datasets.detection_datasets import LISA_DETECTION_DATASET
from Yolo_NAS.datasets.lisa_dataset import LisaDataset

from typing import Dict


# todo: create val dataloder, split the lisa files into train, val, test
@register_dataloader(Dataloaders.LISA_TRAIN_YOLO_NAS)
def coco2017_train_yolo_nas(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_nas_dataset_params",
        dataset_cls=LisaDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )
