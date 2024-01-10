from PIL import Image
import numpy as np
import requests
from super_gradients import init_trainer
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torchvision.transforms as T
import torch
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val
from super_gradients.training.utils.distributed_training_utils import setup_device
import matplotlib.pyplot as plt

if __name__ == '__main__':
    init_trainer()
    setup_device(num_gpus=-1)
    # Load the best model that we trained
    best_model = models.get(Models.RESNET18, num_classes=10,
                            checkpoint_path="/home/matan/models/supergradients/detection/my_cifar_experiment/RUN_20231227_184348_511917/ckpt_best.pth")
    best_model.eval()
    url = "https://www.aquariumofpacific.org/images/exhibits/Magnificent_Tree_Frog_900.jpg"
    image = np.array(Image.open(requests.get(url, stream=True).raw))

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        T.Resize((32, 32))
    ])
    input_tensor = transforms(image).unsqueeze(0).to(next(best_model.parameters()).device)
    predictions = best_model(input_tensor)

    valid_loader = cifar10_val()

    classes = valid_loader.dataset.classes
    plt.xlabel(classes[torch.argmax(predictions)])
    plt.imshow(image)
    plt.show()
