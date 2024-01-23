from typing import Mapping
from super_gradients.common.registry.registry import register_dataloader
from super_gradients.common.object_names import Dataloaders
from super_gradients.training.dataloaders.dataloaders import _process_dataset_params, _process_dataloader_params
from super_gradients.common.environment.ddp_utils import get_local_rank
from super_gradients.common.environment.cfg_utils import load_dataset_params
from super_gradients.training.utils.distributed_training_utils import (
    wait_for_the_master,
)
from super_gradients.training.dataloaders.adapters import maybe_setup_dataloader_adapter
from torch.utils.data import DataLoader
from Yolo_NAS.datasets.lisa_detection import LISADetectionDataset

from typing import Dict
import os


@register_dataloader(Dataloaders.LISA_TRAIN_YOLO_NAS)
def lisa_trafficlight_train_yolo_nas(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_nas_dataset_params",
        dataset_cls=LISADetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.LISA_VAL_YOLO_NAS)
def lisa_trafficlight_val_yolo_nas(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_nas_dataset_params",
        dataset_cls=LISADetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def get_data_loader(config_name: str, dataset_cls: object, train: bool, dataset_params: Mapping = None,
                    dataloader_params: Mapping = None) -> DataLoader:
    """
    Class for creating dataloaders for taking defaults from yaml files in src/super_gradients/recipes.

    :param config_name: yaml config filename of dataset_params in recipes (for example coco_detection_dataset_params).
    :param dataset_cls: torch dataset uninitialized class.
    :param train: controls whether to take
        cfg.train_dataloader_params or cfg.valid_dataloader_params as defaults for the dataset constructor
     and
        cfg.train_dataset_params or cfg.valid_dataset_params as defaults for DataLoader contructor.

    :param dataset_params: dataset params that override the yaml configured defaults, then passed to the dataset_cls.__init__.
    :param dataloader_params: DataLoader params that override the yaml configured defaults, then passed to the DataLoader.__init__
    :return: DataLoader
    """
    if dataloader_params is None:
        dataloader_params = dict()
    if dataset_params is None:
        dataset_params = dict()

    cfg = load_dataset_params(config_name=config_name, recipes_dir_path=os.path.dirname(__file__))

    dataset_params = _process_dataset_params(cfg, dataset_params, train)
    dataset_params['class_names'] = cfg['class_names']

    local_rank = get_local_rank()
    with wait_for_the_master(local_rank):
        dataset = dataset_cls(**dataset_params)
        if not hasattr(dataset, "dataset_params"):
            dataset.dataset_params = dataset_params

    dataloader_params = _process_dataloader_params(cfg, dataloader_params, dataset, train)

    # Ensure there is no dataset in dataloader_params (Could be there if the user provided dataset class name)
    if "dataset" in dataloader_params:
        _ = dataloader_params.pop("dataset")

    dataloader = DataLoader(dataset=dataset, **dataloader_params)
    dataloader.dataloader_params = dataloader_params

    maybe_setup_dataloader_adapter(dataloader=dataloader)
    return dataloader
