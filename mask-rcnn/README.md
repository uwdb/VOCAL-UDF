# MaskRCNN_for_CLEVR_dataset
Pytorch implementation of Mask RCNN on CLEVR dataset.
![alt text](maskrcnn_clevr.png)


## Set up environment



```sh
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
pip install opencv-python
pip install pycocotools
```

## Training a new model

Hyperparameters:

category=True #25 categories or just 1 category)

batch_size = 2

test_split = 5 # how many images for testing set

root = "output" # dataset dir

num_epochs = 10

```sh
python train_clevr.py
```

visualize

```sh
python test_clevr.py
```
Our code is based on the [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) .



