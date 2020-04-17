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


# Download notMNIST dataset
url = 'https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz'

def download_notmnist(url):
    file_path_to_download = 'data/notMNIST_small.tar.gz'
    if not os.path.exists(file_path_to_download):
        print('start loading...')
        urlretrieve(url, file_path_to_download)
        print('...end loading')
    return file_path_to_download


# downloaded_file = download_notmnist(url)


# extract
extracted_dataset_path = 'data/extracted'
def extract(filename):
    print('start extracting...')
    tar = tarfile.open(filename)
    sys.stdout.flush()
    # tar.extractall('data/extracted')
    tar.extractall(extracted_dataset_path)
    tar.close()
    data_folders = sorted(os.listdir(extracted_dataset_path + '/notMNIST_small'))
    print(data_folders)
    return data_folders


# train_folders = extract(downloaded_file)

# show random images from dataset
char_folders_path = extracted_dataset_path + '/notMNIST_small/'

characters = 'abcdefghij'.upper()

list_of_images = []
for idx in range(3):
    for char in characters:
        char_folder = char_folders_path + char + '/'
        images = os.listdir(char_folder)
        image_file_name = images[random.randint(len(images))]
        list_of_images.append(char_folder + image_file_name)


def show__random_images(list_of_files):

    number_of_files = len(list_of_files)
    num_char = len(characters)

    for row in range(int(number_of_files / num_char)):
        fig = figure(figsize=(15, 5))

        for i in range(num_char):
            fig.add_subplot(1, num_char, i + 1)
            image = imread(list_of_files[row * num_char + i])
            plt.imshow(image)
            plt.show(image)
            axis('off')

show__random_images(list_of_images)