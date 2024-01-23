import imagesize
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
from typing import List, Optional, Dict, Union, Any, Tuple
import os
import pandas as pd
import numpy as np
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict


class LisaAnnotationClass(Enum):
    BOX = 0
    BULB = 1
    BOX_PATH = 2
    BULB_PATH = 3


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class LisaFormatDetectionDataset(DetectionDataset):
    def __init__(
            self,
            data_dir: str,
            annotation_dir: str,
            images_dir: str,
            subdirs: str,
            load_image_sizes_lazily: bool = False,
            class_names: List[str] = [],
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
        self.annotation_dir = annotation_dir
        self.subdirs = subdirs
        self.images_dir = images_dir
        self.tight_box_rotation = tight_box_rotation
        self.with_crowd = with_crowd
        self.class_ids_to_ignore = class_ids_to_ignore or []
        self.img_paths_to_sizes: Dict[str, Optional[Tuple]] = {}
        self.img_ids_to_paths: Dict[int, str] = {}
        self.load_image_sizes_lazily = load_image_sizes_lazily
        self.original_classes = class_names

        target_fields = ["target", "crowd_target"] if self.with_crowd else ["target"]
        kwargs["target_fields"] = target_fields
        kwargs["output_fields"] = ["image", *target_fields]
        kwargs["original_target_format"] = XYXY_LABEL
        super().__init__(data_dir=data_dir, *args, **kwargs)

    def _setup_data_source(self) -> int:
        # imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        self.images_dir = os.path.join(self.data_dir, self.images_dir)

        def load_annotations(path: str, annotation_dict_: Dict[str, Dict[LisaAnnotationClass, pd.DataFrame]],
                             subdir: str) -> Dict:
            def read_csv(path):
                df = pd.read_csv(path, delimiter=';')
                df['Filename'] = df['Filename'].apply(lambda path: os.path.basename(path))
                df = df.set_index(['Filename'])
                return df

            if 'lock' in path:
                return
            if not os.path.isfile(path):
                original_subdir = subdir
                for f in os.listdir(path):
                    full_path = os.path.join(path, f)
                    if not os.path.isfile(full_path):
                        subdir = os.path.join(original_subdir, f)
                    load_annotations(os.path.join(path, f), annotation_dict_, subdir)
            else:
                annotation_dict_[subdir] = annotation_dict_.get(subdir, {})
                if LisaAnnotationClass.BOX.name in os.path.basename(path):
                    annotation_dict_[subdir][LisaAnnotationClass.BOX] = read_csv(path)
                elif LisaAnnotationClass.BULB.name in os.path.basename(path):
                    annotation_dict_[subdir][LisaAnnotationClass.BULB] = read_csv(path)
                else:
                    raise Exception('unrecognized file in Lisa dataset')
            return annotation_dict_

        self.annotations: Dict = {}
        for subdir_ in self.subdirs:
            path = os.path.join(self.data_dir, self.annotation_dir, subdir_)
            load_annotations(path, self.annotations, subdir_)

        # get image sizes
        total_imgs = 0
        img_counter = 0
        for i, subdir_ in enumerate(self.subdirs):
            print(f'getting image sizes for subdir {i + 1}/{len(self.subdirs)}')
            paths = [f for f in Path(os.path.join(self.images_dir, subdir_)).rglob('*.jpg')]
            total_imgs += len(paths)
            for f in tqdm(paths, total=len(paths)):
                if not self.load_image_sizes_lazily:
                    width, height = imagesize.get(f)
                    self.img_paths_to_sizes[f] = (width, height)
                else:
                    self.img_paths_to_sizes[f] = None
                self.img_ids_to_paths[img_counter] = f
                img_counter += 1
        return total_imgs

    @property
    def _all_classes(self) -> List[str]:
        return self.original_classes

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        img_path = self.img_ids_to_paths[sample_id]
        assert str(img_path).startswith(self.images_dir)
        suffix_path = os.path.relpath(img_path, self.images_dir)
        suffix_path = suffix_path.split(os.sep)
        if suffix_path[0]==suffix_path[1]:
            suffix_path=suffix_path[1:]
        subdir = '/'.join(suffix_path[:suffix_path.index('frames')])
        img_key = os.path.basename(img_path)

        initial_img_shape = self.img_paths_to_sizes[img_path]
        if initial_img_shape is None:
            initial_img_shape = self.img_paths_to_sizes[img_path] = imagesize.get(img_path)

        annotations = []
        for lisa_annotation_class, df in self.annotations[subdir].items():
            if img_key in df.index:
                ann = df.loc[[img_key]]
                ann['class'] = lisa_annotation_class
                annotations.append(ann)
        if annotations:
            annotations = pd.concat(annotations)

        total_annotations = len(annotations)
        target = np.zeros([total_annotations, 5], np.float32)
        crowd_target = np.zeros([0, 5], np.float32)
        target_segmentation = np.zeros([total_annotations, 0], np.float32)

        if total_annotations > 0:
            for i, row in enumerate(annotations.iterrows()):
                target[i, 0:4] = row[1][['Upper left corner X', 'Upper left corner Y', 'Lower right corner X',
                                         'Lower right corner Y']].values
                target[i, 4] = row[1]['class'].value

        if self.input_dim is not None:
            width, height = initial_img_shape
            r = min(self.input_dim[0] / height, self.input_dim[1] / width)
            target[:, :4] *= r
            crowd_target[:, :4] *= r
            target_segmentation *= r
            resized_img_shape = (int(height * r), int(width * r))
        else:
            resized_img_shape = initial_img_shape

        annotation = {
            "target": target,
            "crowd_target": crowd_target,
            "target_segmentation": target_segmentation,
            "initial_img_shape": initial_img_shape,
            "resized_img_shape": resized_img_shape,
            "img_path": str(img_path),
            "id": np.array([sample_id]),
        }
        return annotation

# matan@matan:~/data/lisa_trafficlight/Images$ cd ./daySequence1/
# matan@matan:~/data/lisa_trafficlight/images/daySequence1$ find . -name "*.jpg" | wc -l
# 4060
# matan@matan:~/data/lisa_trafficlight/images/daySequence1$ cd ../daySequence2
# matan@matan:~/data/lisa_trafficlight/images/daySequence2$ find . -name "*.jpg" | wc -l
# 6894
# matan@matan:~/data/lisa_trafficlight/images/nightTrain$ cd ../nightSequence1
# matan@matan:~/data/lisa_trafficlight/images/nightSequence1$ find . -name "*.jpg" | wc -l
# 4993
# matan@matan:~/data/lisa_trafficlight/images/nightSequence1$ cd ../nightSequence2
# matan@matan:~/data/lisa_trafficlight/images/nightSequence2$ find . -name "*.jpg" | wc -l
# 6534


# matan@matan:~/data/lisa_trafficlight/images/daySequence2$ cd ../dayTrain/
# matan@matan:~/data/lisa_trafficlight/images/dayTrain$ find . -name "*.jpg" | wc -l
# 14034
# matan@matan:~/data/lisa_trafficlight/images/dayTrain$ cd ../nightTrain/
# matan@matan:~/data/lisa_trafficlight/images/nightTrain$ find . -name "*.jpg" | wc -l
# 6501


# matan@matan:~/data/lisa_trafficlight/images/nightSequence2$ cd ../sample-dayClip6/
# matan@matan:~/data/lisa_trafficlight/images/sample-dayClip6$ find . -name "*.jpg" | wc -l
# 468
# matan@matan:~/data/lisa_trafficlight/images/sample-dayClip6$ cd ../sample-nightClip1/
# matan@matan:~/data/lisa_trafficlight/images/sample-nightClip1$ find . -name "*.jpg" | wc -l
# 591
