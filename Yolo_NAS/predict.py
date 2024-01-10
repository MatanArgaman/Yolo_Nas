import super_gradients
import os
from PIL import Image
import requests
from io import BytesIO

from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES

#/home/matan/venvs/torch/lib/python3.8/site-packages/super_gradients/train_from_recipe.py
# add to environment variables: PYTHONPATH = $PYTHONPATH:/home/matan/rep/super-gradients

def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


if __name__ == '__main__':
    # yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()
    yolo_nas = super_gradients.training.models.get("yolox_l", num_classes=PRETRAINED_NUM_CLASSES['coco'], checkpoint_path="/home/matan/rep/super-gradients/checkpoints/yolox_l_coco2017_res[640, 640]/RUN_20240107_165842_599681/ckpt_best.pth").cuda()
    # yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg").show()
    # yolo_nas.predict("https://static.scientificamerican.com/sciam/cache/file/3F302C18-1DA6-4372-8E313E2FF2CF4D86_source.png?w=1350").show()
    dir_path = '/home/matan/data/coco2017/images/val2017'
    # img = get_image("https://i.ebayimg.com/images/g/RIkAAOSwB7FkGEvB/s-l1600.png")
    img = get_image("https://www.amomentwithfranca.com/wp-content/uploads/2017/02/Farm-World-Starter-Set.jpg")
    pred = yolo_nas.predict(img)
    pred.show()
    for f in os.listdir(dir_path):
        yolo_nas.predict(os.path.join(dir_path, f)).show()
