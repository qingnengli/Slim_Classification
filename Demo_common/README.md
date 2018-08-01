# Moderify some codes for D-I-Y datasets

# Create the specified TFRecord from orginal images (.jpg)

# dataset_classification.py Can read all tfrecords, not only cifar10/minist/imagent/flowers

# Adjust train_image_classification.py (train.py) and eval_image_classification.py (eval.py)
# to define num_class, num_sample and so on, but don't need 'dataset_name'

# Training
root@amax:/home/amax/SIAT/Slim_Classification/Demo_common# CUDA_VISIBLE_DEVICES="1" TF_CPP_MIN_LOG_LEVEL="2" python train.py --train_dir=/home/amax/SIAT/Slim_Classification/Demo_common/train --dataset_dir=/home/amax/SIAT/Slim_Classification/Demo_common/dataset/train --num_samples=50000 --num_classes=10 --labels_to_names_path=/home/amax/SIAT/Slim_Classification/Demo_common/dataset/labels.txt --model_name=inception_resnet_v2 --checkpoint_path=/home/amax/SIAT/Slim_Classification/checkpoint/inception_resnet_v2_2016_08_30.ckpt --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --max_number_of_steps=200000

# Validation
root@amax:/home/amax/SIAT/Slim_Classification/Demo_common# CUDA_VISIBLE_DEVICES="0" TF_CPP_MIN_LOG_LEVEL="2" python eval.py --checkpoint_path=/home/amax/SIAT/Slim_Classification/Demo_common/train --eval_dir=/home/amax/SIAT/Slim_Classification/Demo_common/val --dataset_dir=/home/amax/SIAT/Slim_Classification/Demo_common/dataset/test/ --num_samples=10000 --num_classes=10 --model_name=inception_resnet_v2

# Test
root@amax:/home/amax/SIAT/Slim_Classification/Demo_common# CUDA_VISIBLE_DEVICES="0" TF_CPP_MIN_LOG_LEVEL="2" python test.py --checkpoint_path=/home/amax/SIAT/Slim_Classification/Demo_common/train --test_list=/home/amax/SIAT/Slim_Classification/Demo_common/dataset/dataset_test.txt --image_dir=/home/amax/SIAT/Slim_Classification/Data-Cifar10/image --batch_size=32 --num_classes=10 --model_name=inception_resnet_v2

# Tensorboard
tensorboard --logdir ./SIAT/Slim_Classification/Demo_common


