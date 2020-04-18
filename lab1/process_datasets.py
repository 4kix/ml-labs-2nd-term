from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import numpy as np
from numpy import random
import os
import sys
import tarfile
from six.moves.urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from pandas import *
import matplotlib
matplotlib.use('TkAgg')


# Download notMNIST dataset
url_train = 'https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz'
url_test = 'https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz'
train_dataset_name = 'notMNIST_large'
test_dataset_name = 'notMNIST_small'


def download_notmnist(url, path_to_download):
    if not os.path.exists(path_to_download):
        print('start loading...')
        urlretrieve(url, path_to_download)
        print('...end loading')


train_download_path = 'data/notMNIST_large.tar.gz'
test_download_path = 'data/notMNIST_small.tar.gz'
download_notmnist(url_train, train_download_path)
download_notmnist(url_train, test_download_path)

input("Press Enter to continue...")


# extract
extracted_dataset_path = 'data/extracted'


def extract(filename, dataset_name):
    print('start extracting...')
    tar = tarfile.open(filename)
    sys.stdout.flush()
    # tar.extractall('data/extracted')
    tar.extractall(extracted_dataset_path)
    tar.close()
    folders_path = extracted_dataset_path + '/' + dataset_name
    data_folders = sorted(os.listdir(folders_path))
    print(data_folders)
    return data_folders, folders_path


train_folders, train_char_folders_path = extract(train_download_path, train_dataset_name)
test_folders, test_char_folders_path = extract(test_download_path, test_dataset_name)

# train_folders, train_char_folders_path = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 'data/extracted/notMNIST_large'
# test_folders, test_char_folders_path = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 'data/extracted/notMNIST_small'
# char_folders = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

input("Press Enter to continue...")


# show random images from dataset
train_char_folders_path = train_char_folders_path + '/'
test_char_folders_path = test_char_folders_path + '/'

np.random.seed(123)


def show_image_list(list_of_files):
    number_of_ims = len(list_of_files)
    fig = figure(figsize=(number_of_ims, 1))
    for i in range(number_of_ims):
        fig.add_subplot(1, number_of_ims, i + 1)
        image = imread(list_of_files[i])
        imshow(image, cmap='gray')
        axis('off')
    plt.title("Random images from dataset")
    plt.show()

def show_images(char_folders_path, char_folders):
    list_of_images = []
    for i in range(10):
        char = random.choice(char_folders)
        char_folder = char_folders_path + char + '/'
        images = os.listdir(char_folder)
        image_file_name = images[random.randint(len(images))]
        list_of_images.append(char_folder + image_file_name)
    show_image_list(list_of_images)


show_images(train_char_folders_path, train_folders)
show_images(test_char_folders_path, test_folders)

input("Press Enter to continue...")


# check if dataset is balanced
def check_balance(char_folders_path, char_folders):
    dataset_dic = dict()
    im_total = 0
    for char in char_folders:
        char_folder = char_folders_path + char + '/'
        img_num = len(os.listdir(char_folder))
        dataset_dic[char] = img_num
        im_total += img_num

    print(dataset_dic)
    plt.bar(list(dataset_dic.keys()), dataset_dic.values(), color='g')
    plt.title("Class balance in set")
    plt.show()
    return im_total


train_im_total = check_balance(train_char_folders_path, train_folders)
test_im_total = check_balance(test_char_folders_path, test_folders)

input("Press Enter to continue...")

# Prepare datasets



def parse_dataset(char_folders_path, char_folders, im_total):
    X = np.zeros((im_total, 28, 28))
    y = np.zeros(im_total)
    arr_counter = 0
    for i, char in enumerate(char_folders):
        char_folder = char_folders_path + char + '/'
        images = os.listdir(char_folder)
        for im_idx, image_name in enumerate(images):
            try:
                X[arr_counter, :, :] = imread(char_folder + image_name)
                y[arr_counter] = i
                arr_counter += 1
            except:
                # when imread fails because there are corrupted/empty files in dataset
                continue
    return X, y

print("Start parsing...")
X_train, y_train = parse_dataset(train_char_folders_path, train_folders, train_im_total)
X_test, y_test = parse_dataset(test_char_folders_path, test_folders, test_im_total)
print("...end parsing arrays")

input("Press Enter to continue...")


# Merge and split into train, test, validation sets
print("Start merging arrays...")
# X_train = np.concatenate((X_train, X_test), 0)
# y_train = np.concatenate((y_train, y_test), 0)
print("...end merging arrays")

input("Press Enter to continue...")

# X_train = np.load('X_merged.npy')
# y_train = np.load('y_merged.npy')

train_size = 200000
test_size = 19000
val_size = 10000

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=train_size)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size, train_size=val_size)

print("X_train size: {}, y_train size: {}".format(X_train.shape[0], y_train.shape[0]))
print("X_test size: {}, y_test size: {}".format(X_test.shape[0], y_test.shape[0]))
print("X_val size: {}, y_val size: {}".format(X_val.shape[0], y_val.shape[0]))

input("Press Enter to continue...")

# Remove duplicates


def remove_duplicates(set_x_1, set_x_2, set_y_1):
    hash_tab = {}
    duplicate_indexes = []
    duplicate_counter = 0
    for i, img in enumerate(set_x_1):
        base_img_hash = hash(bytes(img))
        hash_tab[base_img_hash] = i
    for j, img2 in enumerate(set_x_2):
        img_to_cmp_hash = hash(bytes(img2))
        if img_to_cmp_hash in hash_tab:
            duplicate_indexes.append(hash_tab[img_to_cmp_hash])
            duplicate_counter += 1
    set_x_1 = np.delete(set_x_1, duplicate_indexes, 0)
    set_y_1 = np.delete(set_y_1, duplicate_indexes, 0)
    print("Found and deleted {} duplicates".format(duplicate_counter))
    return set_x_1, set_y_1


X_train, y_train = remove_duplicates(X_train, X_test, y_train)
X_train, y_train = remove_duplicates(X_train, X_val, y_train)

input("Press Enter to continue...")

# Save datasets to *.NPY files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)

input("Press Enter to continue...")

