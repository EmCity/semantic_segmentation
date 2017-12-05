# semantic_segmentation
My solution for semantic segmentation task on MSRC-v2 Segmentation Dataset

## Download the dataset
```shell
$ ./download_datasets.sh
```

## Usage
```shell
$ python main.py [-pm=] [-e=]
```
- __pm:__ choice of pre-trained models from vgg11, vgg16 and vgg19, vgg11 by default
- __e:__ Number of epoch, i.e. number of iteration of training, 10 by default
