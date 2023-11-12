import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from classes.siamese_model import SiameseModel
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from utils.data_preparation_utils import (create_batch, create_triplets,
                                          split_dataset)

tf.__version__, np.__version__
tf.random.set_seed(5)

model = tf.keras.models.load_model('saved_models/model_v1')

model.summary()

ROOT = "./data/Extracted Faces/Extracted Faces"

train_list, test_list = split_dataset(ROOT, split=0.9)

train_triplet = create_triplets(ROOT, train_list)
test_triplet = create_triplets(ROOT, test_list)

pos_list = np.array([])
neg_list = np.array([])


def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = model.encoder.predict(face_list1)
    tensor2 = model.encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


num_plots = 6
f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))

for data in create_batch(ROOT, test_triplet, batch_size=256):
    a, p, n = data

    pos_class = classify_images(a, p)
    neg_class = classify_images(a, n)
    pos_list = np.append(pos_list, pos_class)
    neg_list = np.append(neg_list, neg_class)
    break

true = np.array([0]*len(pos_list)+[1]*len(neg_list))
pred = np.append(pos_list, neg_list)

print(f"\nAccuracy of model: {accuracy_score(true, pred)}\n")


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
    plt.title('Means Comparision')
    plt.legend()

    ap_75quartile = (ap_mean+ap_stds)
    an_75quartile = (an_mean-an_stds)
    plt.subplot(122)
    plt.plot(ap_75quartile, 'b', label='AP (Mean+SD)')
    plt.plot(an_75quartile, 'g', label='AN (Mean-SD)')
    plt.title('75th Quartile Comparision')
    plt.legend()


def ModelMetrics(pos_list, neg_list):
    true = np.array([0]*len(pos_list)+[1]*len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred)}\n")

    # Compute and plot the Confusion matrix
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

plot_metrics(train_loss, test_metrics)
ModelMetrics()
