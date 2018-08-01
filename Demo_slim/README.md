# Keep all files in model/research/slim packages
# Only suitable to cifar10/mnist/flowers/imagenet
# Urge to modefity some codes for my own datasets


# Training
root@amax:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim# CUDA_VISIBLE_DEVICES="1" TF_CPP_MIN_LOG_LEVEL="2" python train_image_classifier.py --dataset_name=cifar10 --dataset_dir=/home/amax/SIAT/Slim_Classification/Data-Cifar10/tfrecord --checkpoint_path=/home/amax/SIAT/Slim_Classification/checkpoint/inception_resnet_v2_2016_08_30.ckpt --model_name=inception_resnet_v2 --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --train_dir=/home/amax/SIAT/Slim_Classification/Demo_slim/train 

#Validation
root@amax:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim# CUDA_VISIBLE_DEVICES="0" TF_CPP_MIN_LOG_LEVEL="2" python  eval_image_classifier.py --dataset_name=cifar10 --dataset_dir=/home/amax/SIAT/Slim_Classification/Data-Cifar10/tfrecord --dataset_split_name=test --model_name=inception_resnet_v2 --checkpoint_path=/home/amax/SIAT/Slim_Classification/Demo_slim/train --eval_dir=/home/amax/SIAT/Slim_Classification/Demo_slim/val

# Tensorboard
tensorboard --logdir ./SIAT/Slim_Classification/Demo_slim

# Frozen graph
root@amax:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim# python export_inference_graph.py --model_name=inception_resnet_v2 --output_file=/home/amax/SIAT/Slim_Classification/Demo_slim/train/frozen_inception_resnet_v2.pb --dataset_name=cifar10 --dataset_dir=/home/amax/SIAT/Slim_Classification/Data-Cifar10/tfrecord

