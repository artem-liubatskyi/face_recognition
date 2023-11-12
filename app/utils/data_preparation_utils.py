import os
import random

import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input

random.seed(5)
np.random.seed(5)


def read_image(index, directory):
    path = os.path.join(directory, index[0], index[1])
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def split_dataset(directory, split=0.9):
    folders = os.listdir(directory)
    num_train = int(len(folders)*split)
    random.shuffle(folders)

    return folders[:num_train], folders[num_train:]


def create_triplets(directory, folder_list, max_files=10):
    triplets = []

    for folder in folder_list:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)

        for i in range(num_files-1):
            for j in range(i + 1, num_files):
                anchor = (folder, f"{i}.jpg")
                positive = (folder, f"{j}.jpg")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folder_list)
                negative = (
                    neg_folder, f"{random.randint(0, len(list(os.listdir(os.path.join(directory, neg_folder)))[:max_files]) - 1)}.jpg")

                triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets


def create_batch(directory, triplet_list, batch_size=256, preprocess=True):
    batches_count = len(triplet_list)//batch_size

    for i in range(batches_count + 1):
        anchor = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            anchor.append(read_image(a, directory))
            positive.append(read_image(p, directory))
            negative.append(read_image(n, directory))
            j += 1

        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([anchor, positive, negative])
