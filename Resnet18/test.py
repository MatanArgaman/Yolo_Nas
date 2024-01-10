from super_gradients import init_trainer
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_val
from super_gradients.training.utils.distributed_training_utils import setup_device
if __name__ == '__main__':

    init_trainer()
    setup_device(num_gpus=-1)
    trainer = Trainer(experiment_name="test_my_cifar_experiment", ckpt_root_dir="/home/matan/models/supergradients/detection/my_cifar_experiment/RUN_20231227_184348_511917")
    model = models.get(Models.RESNET18, num_classes=10,
                       checkpoint_path="/home/matan/models/supergradients/detection/my_cifar_experiment/RUN_20231227_184348_511917/ckpt_best.pth")
    test_metrics = [Accuracy(), Top5()]
    test_data_loader = cifar10_val()
    test_results = trainer.test(model=model, test_loader=test_data_loader, test_metrics_list=test_metrics)
    print(f"Test results: Accuracy: {test_results['Accuracy']}, Top5: {test_results['Top5']}")

