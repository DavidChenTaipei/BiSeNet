## Pretrained Weights (Tain by Me)
Pretrained can be downloaded from [here](https://drive.google.com/drive/folders/1Litp2dKYYTB2z9H7yw7fSLP4oqTNLTV0?usp=sharing)
## platform
My platform is like this: 
* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.51.05
* cuda 10.2
* cudnn 7
* miniconda python 3.6.9
* pytorch 1.6.0


## get start
With a pretrained weight, you can run inference on an single image like this: 
```
$ python tools/demo.py --model bisenetv2 --weight-path /path/to/your/weights.pth --img-path ./example.png
```
This would run inference on the image and save the result image to `./res.jpg`.


## prepare dataset

cityscapes  

Register and download the dataset from the official [website](https://www.cityscapes-dataset.com/). Then decompress them into the `datasets/cityscapes` directory:  
```
$ mv /path/to/leftImg8bit_trainvaltest.zip datasets/cityscapes
$ mv /path/to/gtFine_trainvaltest.zip datasets/cityscapes
$ cd datasets/cityscapes
$ unzip leftImg8bit_trainvaltest.zip
$ unzip gtFine_trainvaltest.zip
```

## train
In order to train the model, you can run command like this: 
```
$ export CUDA_VISIBLE_DEVICES=0,1

# if you want to train with apex
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --model bisenetv2 # or bisenetv1

# if you want to train with pytorch fp16 feature from torch 1.6
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --model bisenetv2 # or bisenetv1
```

## eval pretrained models
You can also evaluate a trained model like this: 
```
$ python tools/evaluate.py --model bisenetv1 --weight-path /path/to/your/weight.pth
```
