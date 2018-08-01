# coding=utf-8
import os
import random
# label.txt
# 0:airplane
# 1:automobile
# 2:bird
# 3:cat
# 4:deer
# 5:dog
# 6:frog
# 7:horse
# 8:ship
# 9:truck

def generate_dataset_list(dataset_name):
    class_names_to_ids = {0:'airplane', 1: 'automobile', 2: 'bird',
                          3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                          7: 'horse', 8: 'ship', 9: 'truck'}
    data_dir = '/home/amax/SIAT/Slim_Classification/Data-Cifar10/image/'+dataset_name + '/'
    output_path = 'dataset_' + dataset_name +'.txt'
    fd = open(output_path, 'w')
    # for class_name in class_names_to_ids.keys():
    for i in range(10):
        class_name = class_names_to_ids[i]
        images_list = os.listdir(data_dir + str(i))
        for image_name in images_list:
            fd.write('{}/{}/{} {}\n'.format(dataset_name, i, image_name, i))
    fd.close()

def split_data(dataset_list):
    _NUM_VALIDATION = 350
    _RANDOM_SEED = 0
    dataset_train = 'dataset_train.txt'
    dataset_test = 'dataset_test.txt'
    fd = open(dataset_list)
    lines = fd.readlines()
    fd.close()
    random.seed(_RANDOM_SEED)
    random.shuffle(lines)
    fd = open(dataset_train, 'w')
    for line in lines[_NUM_VALIDATION:]:
        fd.write(line)
    fd.close()
    fd = open(dataset_test, 'w')
    for line in lines[:_NUM_VALIDATION]:
        fd.write(line)
    fd.close()

if __name__ == '__main__':
    label_train= generate_dataset_list('train')
    label_test = generate_dataset_list('test')
