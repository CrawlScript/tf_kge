# coding=utf-8
import tensorflow as tf
import numpy as np


class TransE(tf.keras.Model):

    def __init__(self, num_entities, num_relations, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.entity_embeddings = tf.Variable(
            tf.random.truncated_normal([num_entities, embedding_size], stddev=np.sqrt(1 / embedding_size))
        )
        self.relation_embeddings = tf.Variable(
            tf.random.truncated_normal([num_relations, embedding_size], stddev=np.sqrt(1 / embedding_size))
        )

    def embed_norm(self, embeddings, indices):

        # if embedding table is smaller, normalizing first is more efficient
        norm_first = embeddings.shape[0] < len(indices)

        if norm_first:
            h = tf.math.l2_normalize(embeddings, axis=-1)
        else:
            h = embeddings

        h = tf.nn.embedding_lookup(h, indices)

        if not norm_first:
            h = tf.math.l2_normalize(h, axis=-1)

        return h

    def embed_norm_entities(self, entities):
        return self.embed_norm(self.entity_embeddings, entities)

    def embed_norm_relations(self, relations):
        return self.embed_norm(self.relation_embeddings, relations)

    def call(self, inputs, target_entity_type, training=None, mask=None):
        """

        :param inputs: [source_entities, relations]
        :param target_entity_type: "head" | "tail"
        :param training:
        :param mask:
        :return:
        """
        source, r = inputs

        embedded_source = self.embed_norm_entities(source)
        embedded_r = self.embed_norm_relations(r)

        if target_entity_type == "tail":
            translated = embedded_source + embedded_r
        else:
            translated = embedded_source - embedded_r

        return translated
