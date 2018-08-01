# Slim_Classification
Deep learning Classification Task on TFSlim tools 

## Dataset
Cifar10: offical downloaded files (.meta and data_batch1-5), generated TFrecords files and original images (32*32, 10 classes) which are recovered by myself. 

## Checkpoint
The chechpoint folder contains the pretrained model for slim tools--inception_resnet_v2.ckpt

## Three Methods
Do Classification Task in Cifar10 data with three models:
1. Build-in API in tensorflow/models/research/slim. download_convert_cifar10.py and so on
2. Modified the slim api so as to extend to other similar datasets, inculding gennerate_tfrecords, read_tfrecord and train.
3. Self-multigpu model without any pretrained models to train.
