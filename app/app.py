import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from classes.siamese_model import SiameseModel
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.optimizers import Adam
from utils.data_preparation_utils import (create_batch, create_triplets,
                                          split_dataset)

DIRECTORY = "./data/Extracted Faces/Extracted Faces"

train_list, test_list = split_dataset(DIRECTORY)
train_triplet = create_triplets(DIRECTORY, train_list)
test_triplet = create_triplets(DIRECTORY, test_list)

num_plots = 6
f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))

for x in create_batch(DIRECTORY, train_triplet, batch_size=num_plots, preprocess=False):
    a, p, n = x
    for i in range(num_plots):
        axes[i, 0].imshow(a[i])
        axes[i, 1].imshow(p[i])
        axes[i, 2].imshow(n[i])
        i += 1
    break

plt.show()

siamese_model = SiameseModel()
siamese_model.compile(optimizer=Adam(learning_rate=1e-3, epsilon=1e-01))

epochs = 10
batch_size = 128

max_acc = 0
train_loss = []
test_metrics = []

for epoch in range(1, epochs + 1):
    epoch_loss = []
    for data in create_batch(DIRECTORY, train_triplet, batch_size):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    train_loss.append(epoch_loss)
    metric = siamese_model.test_on_triplets(DIRECTORY, test_triplet, batch_size)
    test_metrics.append(metric)
    accuracy = metric[0]

    if accuracy >= max_acc:
        siamese_model.save_weights("siamese_model")
        max_acc = accuracy

siamese_model.save_weights("siamese_model")
siamese_model.save('saved_models/model_v1')

test_metrics = np.array(test_metrics)

siamese_model.encoder.save_weights("encoder")
siamese_model.encoder.summary()


def plot_metrics(loss, metrics):
    accuracy = metrics[:, 0]
    ap_mean = metrics[:, 1]
    an_mean = metrics[:, 2]
    ap_stds = metrics[:, 3]
    an_stds = metrics[:, 4]

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(loss, 'b', label='Loss')
    plt.title('Training loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(accuracy, 'r', label='Accuracy')
    plt.title('Testing Accuracy')
    plt.legend()

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(ap_mean, 'b', label='AP Mean')
    plt.plot(an_mean, 'g', label='AN Mean')
    plt.title('Means Comparison')
    plt.legend()

    ap_75quartile = (ap_mean+ap_stds)
    an_75quartile = (an_mean-an_stds)
    plt.subplot(122)
    plt.plot(ap_75quartile, 'b', label='AP (Mean+SD)')
    plt.plot(an_75quartile, 'g', label='AN (Mean-SD)')
    plt.title('75th Quartile Comparison')
    plt.legend()


def plot_confusion_matrix():
    pos_list = np.array([])
    neg_list = np.array([])

    for data in create_batch(DIRECTORY, test_triplet, batch_size=256):
        a, p, n = data
        pos_list = np.append(pos_list, siamese_model.classify_images(a, p))
        neg_list = np.append(neg_list, siamese_model.classify_images(a, n))
        break
    true = np.array([0]*len(pos_list)+[1]*len(neg_list))
    pred = np.append(pos_list, neg_list)

    cf_matrix = confusion_matrix(true, pred)

    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar',
             'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(
        value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()
    cv2.waitKey(0)


plot_metrics(train_loss, test_metrics)
plot_confusion_matrix()
