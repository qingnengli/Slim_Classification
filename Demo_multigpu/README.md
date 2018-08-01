# tensorflow_multigpu_imagenet
Code for training different architectures( DenseNet, ResNet, AlexNet, GoogLeNet, VGG, NiN) on ImageNet or other datasets + Multi-GPU support + Transfer Learning support

This repository provides an easy-to-use way for training different well-known deep learning architectures on different datasets.

Moreover, multi-GPU and transfer learning are supported.

This code takes advantage of these repositories:

https://github.com/soumith/imagenet-multiGPU.torch

https://github.com/ry/tensorflow-resnet

https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10


The code reads dataset information from a text or csv file and directly loads images from disk.




#Example of usages:


# Training:
# --log_dir/num_sample/num_batches is not neccessary
# --retrain_from: ckpt_dir. Desire transfer_mode[0]=1 or 3.(transfer_mode=[1],[3,Tuning_epoch])

CUDA_VISIBLE_DEVICES="1" TF_CPP_MIN_LOG_LEVEL="2" python train.py --architecture densenet --depth 121 --path_prefix /home/amax/SIAT/Slim_Classification/Data-Cifar10/image/ --data_info dataset_train.txt --snapshot_prefix "Cifar10_densenet" --log_debug_info True --max_to_keep 3 --num_epochs 100 --batch_size 16 --num_classes 10 --log_dir "Densenet_Log"

# Evaluating a trained model:


CUDA_VISIBLE_DEVICES="0" TF_CPP_MIN_LOG_LEVEL="2" python eval.py --architecture densenet --depth 121 --log_dir "Densenet_Log" --path_prefix /home/amax/SIAT/Slim_Classification/Data-Cifar10/image/ --data_info dataset_test.txt --num_classes 10 --save_predictions "eval_metric_Topacc.txt" --top_n 3



# Transfer learning:

python train.py --transfer_mode 1 --architecture alexnet --retrain_from ./alexnet_Run-17-07-2017-15:31:57
