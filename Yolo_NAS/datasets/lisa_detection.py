import os

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.abstractions.abstract_logger import get_logger
from typing import List

from Yolo_NAS.datasets.lisa_format_detection import LisaFormatDetectionDataset

logger = get_logger(__name__)


@register_dataset(Datasets.LISA_DETECTION_DATASET)
class LISADetectionDataset(LisaFormatDetectionDataset):
    """Dataset for COCO object detection.

    To use this Dataset you need to:

        - Download coco dataset:
            annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
            train2017: http://images.cocodataset.org/zips/train2017.zip
            val2017: http://images.cocodataset.org/zips/val2017.zip

        - Unzip and organize it as below:
            coco
            ├── annotations
            │      ├─ instances_train2017.json
            │      ├─ instances_val2017.json
            │      └─ ...
            └── images
                ├── train2017
                │   ├─ 000000000001.jpg
                │   └─ ...
                └── val2017
                    └─ ...

        - Install CoCo API: https://github.com/pdollar/coco/tree/master/PythonAPI

        - Instantiate the dataset:
            >> train_set = COCODetectionDataset(data_dir='.../coco', subdir='images/train2017', json_file='instances_train2017.json', ...)
            >> valid_set = COCODetectionDataset(data_dir='.../coco', subdir='images/val2017', json_file='instances_val2017.json', ...)
    """

    def __init__(
            self,
            annotations_dir: str = "annotations",
            images_dir: str = "images",
            json_file="",  # removes 'json_file' from kwargs
            subdirs: str = "images",
            load_image_sizes_lazily: bool = False,
            class_names: List[str] = [],
            *args,
            **kwargs,
    ):
        """
        :param json_file:           Name of the coco json file, that resides in data_dir/annotations/json_file.
        :param subdir:              Sub directory of data_dir containing the data.
        :param tight_box_rotation:  bool, whether to use of segmentation maps convex hull as target_seg
                                    (check get_sample docs).
        :param with_crowd: Add the crowd groundtruths to __getitem__

        kwargs:
            all_classes_list: all classes list, default is COCO_DETECTION_CLASSES_LIST.
        """

        super().__init__(annotation_dir=annotations_dir, images_dir=images_dir, subdirs=subdirs,
                         load_image_sizes_lazily=load_image_sizes_lazily, class_names=class_names,
                         *args, **kwargs)
