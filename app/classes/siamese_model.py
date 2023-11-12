import numpy as np
import tensorflow as tf
from classes.triplet_loss_layer import TripletLossLayer
from tensorflow.keras import layers, metrics
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential
from utils.data_preparation_utils import create_batch


def create_encoder(input_shape):
    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for i in range(len(pretrained_model.layers)-27):
        pretrained_model.layers[i].trainable = False

    return Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")


class SiameseModel(Model):
    def __init__(self, input_shape=(128, 128, 3), margin=1.0):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.encoder = create_encoder(input_shape)
        self.siamese_network = self.create_siamese_network(
            self.encoder, input_shape)
        self.siamese_network.summary()
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.siamese_network.trainable_weights), self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    def classify_images(self, face_list1, face_list2, threshold=1.3):
        tensor1 = self.encoder.predict(face_list1)
        tensor2 = self.encoder.predict(face_list2)

        distance = np.sum(np.square(tensor1-tensor2), axis=-1)
        prediction = np.where(distance <= threshold, 0, 1)
        return prediction

    def test_on_triplets(self, directory, triplets, batch_size=256):
        pos_scores, neg_scores = [], []

        for data in create_batch(directory, triplets, batch_size):
            prediction = self.predict(data)
            pos_scores += list(prediction[0])
            neg_scores += list(prediction[1])

        accuracy = np.sum(np.array(pos_scores) < np.array(
            neg_scores)) / len(pos_scores)
        ap_mean = np.mean(pos_scores)
        an_mean = np.mean(neg_scores)
        ap_stds = np.std(pos_scores)
        an_stds = np.std(neg_scores)

        print(f"Accuracy on test = {accuracy:.5f}")
        return (accuracy, ap_mean, an_mean, ap_stds, an_stds)

    def create_siamese_network(encoder, input_shape=(128, 128, 3)):
        anchor_input = layers.Input(input_shape, name="Anchor_Input")
        positive_input = layers.Input(input_shape, name="Positive_Input")
        negative_input = layers.Input(input_shape, name="Negative_Input")

        siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=TripletLossLayer()(
                encoder(anchor_input),
                encoder(positive_input),
                encoder(negative_input)
            ),
            name="Siamese_Network"
        )
        return siamese_network

    @property
    def metrics(self):
        return [self.loss_tracker]
