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

    def embed_norm(self, embeddings, indices, norm=True):
        embedded = tf.nn.embedding_lookup(embeddings, indices)
        if norm:
            embedded = tf.math.l2_normalize(embedded, axis=-1)
        return embedded

    def embed_norm_entities(self, entities, norm=True):
        return self.embed_norm(self.entity_embeddings, entities, norm=norm)

    def embed_norm_relations(self, relations, norm=True):
        return self.embed_norm(self.relation_embeddings, relations, norm=norm)

    def call(self, inputs, target_entity_type, training=None, mask=None):
        batch_source, batch_r = inputs

        embedded_source = self.embed_norm_entities(batch_source)
        embedded_r = self.embed_norm_relations(batch_r)

        if target_entity_type == "tail":
            translated = embedded_source + embedded_r
        else:
            translated = embedded_source - embedded_r

        return translated
