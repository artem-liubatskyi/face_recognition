import tensorflow as tf
from tensorflow.keras import layers


class TripletLossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        positive_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        negative_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (positive_distance, negative_distance)
