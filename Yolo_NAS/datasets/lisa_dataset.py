import imagesize
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
from typing import List, Optional, Dict, Union, Any
import os
import pandas as pd
import numpy as np
from enum import Enum
from tqdm import tqdm


class LisaAnnotation(Enum):
    BOX = 0,
    BULB = 1,
    BOX_PATH = 2,
    BULB_PATH = 3


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class LisaDataset(DetectionDataset):
    def __init__(
            self,
            data_dir: str,
            annotation_dir: str,
            images_dir: str,
            tight_box_rotation: bool = False,
            with_crowd: bool = True,
            class_ids_to_ignore: Optional[List[int]] = None,
            *args,
            **kwargs,
    ):
        """
        :param data_dir:                Where the data is stored.
        :param json_annotation_file:    Name of the coco json file. Path relative to data_dir.
        :param images_dir:              Name of the directory that includes all the images. Path relative to data_dir.
        :param tight_box_rotation:      bool, whether to use of segmentation maps convex hull as target_seg
                                            (check get_sample docs).
        :param with_crowd:              Add the crowd groundtruths to __getitem__
        :param class_ids_to_ignore:     List of class ids to ignore in the dataset. By default, doesnt ignore any class.
        """
        self.images_dir = images_dir
        self.annotation_dir = annotation_dir
        self.tight_box_rotation = tight_box_rotation
        self.with_crowd = with_crowd
        self.class_ids_to_ignore = class_ids_to_ignore or []

        target_fields = ["target", "crowd_target"] if self.with_crowd else ["target"]
        kwargs["target_fields"] = target_fields
        kwargs["output_fields"] = ["image", *target_fields]
        kwargs["original_target_format"] = XYXY_LABEL
        super().__init__(data_dir=data_dir, *args, **kwargs)

    def _setup_data_source(self) -> int:

        # imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        def load_annotations(path: str, annotation_dict: Dict) -> Dict:
            if not os.path.isfile(path):
                for f in os.listdir(path):
                    load_annotations(os.path.join(path, f))
            else:
                annotation_dict[path] = annotation_dict.get(path,
                                                            {LisaAnnotation.BOX: '', LisaAnnotation.BULB: '',
                                                             LisaAnnotation.BOX_PATH: '', LisaAnnotation.BULB_PATH: ''})
                if LisaAnnotation.BOX in path:
                    annotation_dict[path][LisaAnnotation.BOX_PATH] = path
                    annotation_dict[LisaAnnotation.BOX] = pd.read_csv(path, delimiter=';')
                elif LisaAnnotation.BULB in path:
                    annotation_dict[path][LisaAnnotation.BULB_PATH] = path
                    annotation_dict[LisaAnnotation.BULB] = pd.read_csv(path, delimiter=';')
                else:
                    raise Exception('unrecognized file in Lisa dataset')

        self.annotations: Dict = load_annotations(self.annotation_dir, {})

        # add image sizes
        for path, annotation_dict in tqdm(self.annotations.items(), total=len(self.annotations.keys())):
            for annotation_file_type in [LisaAnnotation.BOX, LisaAnnotation.BULB]:
                annotation_file_ = annotation_dict[annotation_file_type]
                annotation_dict['img_height'] = pd.Series(dtype='int')
                annotation_dict['img_width'] = pd.Series(dtype='int')
                for index, row in annotation_file_.iterrows():
                    file_path = os.path.join(self.images_dir, row['Origin file'])
                    width, height = imagesize.get(file_path)
                    row['img_height'] = height
                    row['img_width'] = width

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        pass

        # initial_img_shape = (height, width)

        # if self.input_dim is not None: #todo: make sure input_dim is inherited and is 640,640
        #     r = min(self.input_dim[0] / height, self.input_dim[1] / width)
        #     target[:, :4] *= r
        #     crowd_target[:, :4] *= r
        #     target_segmentation *= r
        #     resized_img_shape = (int(height * r), int(width * r))
        # else:
        #     resized_img_shape = initial_img_shape
        #
        # annotation = {
        #     "target": target,
        #     "crowd_target": crowd_target,
        #     "target_segmentation": target_segmentation,
        #     "initial_img_shape": initial_img_shape,
        #     "resized_img_shape": resized_img_shape,
        #     "img_path": img_path,
        #     "id": np.array([img_id]),
        # }
