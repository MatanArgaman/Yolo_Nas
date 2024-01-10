from super_gradients import init_trainer
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import coco2017_val_yolo_nas, coco2017_train_yolo_nas
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, \
    coco_detection_yolo_format_val
import torch
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

from coco_classes import coco_classes


def detection_metric_050_factory():
    return DetectionMetrics_050(num_cls=PRETRAINED_NUM_CLASSES['coco'],
                                score_thres=0.1,
                                top_k_predictions=300,
                                normalize_targets=True,
                                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                       nms_top_k=1000,
                                                                                       max_predictions=300,
                                                                                       nms_threshold=0.7))


if __name__ == '__main__':
    init_trainer()
    setup_device(num_gpus=-1)
    CHECKPOINT_DIR = "/home/matan/models/supergradients/yolo_nas_detection"

    trainer = Trainer(experiment_name='my_first_yolonas_run', ckpt_root_dir=CHECKPOINT_DIR)

    dataset_params = {
        'data_dir': '/home/matan/data/coco2017',
        'train_images_dir': 'images/train2017',
        'train_labels_dir': 'annotations/train',
        'val_images_dir': 'images/val2017',
        'val_labels_dir': 'annotations/val',
        'test_images_dir': 'images/val2017',
        'test_labels_dir': 'annotations/val',
        'classes': coco_classes
    }

    # print(train_data.dataset.transforms)

    model = models.get(Models.YOLO_NAS_L, num_classes=PRETRAINED_NUM_CLASSES['coco'], pretrained_weights='coco')

    train_loader = coco2017_train_yolo_nas(
        dataset_params={'data_dir': '/home/matan/data/coco2017/', 'cache_annotations': False,
                        'ignore_empty_annotations': False},
        dataloader_params={'num_workers': 2, 'batch_size': 8, 'drop_last': True, 'shuffle':True})
    valid_loader = coco2017_val_yolo_nas(
        dataset_params={'data_dir': '/home/matan/data/coco2017/', 'cache_annotations': False,
                        'ignore_empty_annotations': False},
        dataloader_params={'num_workers': 2, 'batch_size': 8, 'drop_last': True, 'shuffle':True})

    train_params = {
        # ENABLING SILENT MODE
        'silent_mode': False,
        "average_best_models": False,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
        "max_epochs": 10,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            # NOTE: num_classes needs to be defined here
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                # NOTE: num_classes needs to be defined here
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    trainer.train(model=model,
                  training_params=train_params,
                  train_loader=train_loader,
                  valid_loader=valid_loader)

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # model = models.get(Models.YOLO_NAS_L, num_classes=PRETRAINED_NUM_CLASSES['coco'], pretrained_weights='coco')
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # training_params = {
    #     "max_epochs": 20,
    #     "initial_lr": 4e-4,
    #     "loss": PPYoloELoss(num_classes=PRETRAINED_NUM_CLASSES['coco']),
    #     "train_metrics_list": [detection_metric_050_factory()],
    #     "valid_metrics_list": [detection_metric_050_factory()],
    #     "metric_to_watch": 'mAP@0.50',
    #     "greater_metric_to_watch_is_better": True,
    # }
    # # 'PPYoloELoss/loss_cls', 'PPYoloELoss/loss_iou', 'PPYoloELoss/loss_dfl', 'PPYoloELoss/loss', 'Precision@0.50', 'Recall@0.50', 'mAP@0.50', 'F1@0.50'
    #
    # train_loader = coco2017_train_yolo_nas(
    #     dataset_params={'data_dir': '/home/matan/data/coco2017/', 'cache_annotations': False,
    #                     'ignore_empty_annotations': False},
    #     dataloader_params={'num_workers': 2, 'batch_size': 3})
    # valid_loader = coco2017_val_yolo_nas(
    #     dataset_params={'data_dir': '/home/matan/data/coco2017/', 'cache_annotations': False,
    #                     'ignore_empty_annotations': False},
    #     dataloader_params={'num_workers': 2})
    #
    # trainer.train(model=model, training_params=training_params, train_loader=train_loader, valid_loader=valid_loader)
