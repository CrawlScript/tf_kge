# coding=utf-8
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tf_kge

from tqdm import tqdm
import tensorflow as tf
from tf_kge.dataset.wn18 import WN18Dataset
from tf_kge.utils.sampling_utils import entity_negative_sampling
import numpy as np

train_kg, test_kg, valid_kg, entity_indexer, relation_indexer = WN18Dataset().load_data()

embedding_size = 20
entity_embeddings = tf.Variable(tf.random.truncated_normal([train_kg.num_entities, embedding_size], stddev=np.sqrt(1 / embedding_size)))
relation_embeddings = tf.Variable(tf.random.truncated_normal([train_kg.num_relations, embedding_size], stddev=np.sqrt(1 / embedding_size)))


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

def compute_distance(a, b):
    # return tf.reduce_sum((a - b) ** 2, axis=-1)
    return tf.reduce_sum(tf.abs(a - b), axis=-1)


for epoch in range(10000):

    for step, (batch_h, batch_r, batch_t) in tqdm(enumerate(tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).shuffle(10000).batch(1000))):
        for target_entity_type in ["head", "tail"]:
            with tf.GradientTape() as tape:
                if target_entity_type == "tail":
                    batch_source = batch_h
                    batch_target = batch_t
                else:
                    batch_source = batch_t
                    batch_target = batch_h


                batch_neg_target = entity_negative_sampling(batch_source, batch_r, kg=train_kg, target_entity_type=target_entity_type, filtered=True)

                embedded_source = tf.nn.embedding_lookup(entity_embeddings, batch_source)
                embedded_r = tf.nn.embedding_lookup(relation_embeddings, batch_r)
                embedded_target = tf.nn.embedding_lookup(entity_embeddings, batch_target)
                embedded_neg_target = tf.nn.embedding_lookup(entity_embeddings, batch_neg_target)

                if target_entity_type == "tail":
                    translated = embedded_source + embedded_r
                else:
                    translated = embedded_source - embedded_r

                margin = 2.0
                losses = tf.maximum(
                    margin + compute_distance(translated, embedded_target) - compute_distance(translated, embedded_neg_target), 0.0)

                loss = tf.reduce_mean(losses)

            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

            if step % 200 == 0:
                print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))


    mean_ranks = []
    for test_step, (batch_h, batch_r, batch_t) in tqdm(enumerate(tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(20))):
        batch_source = batch_h
        batch_target = batch_t

        embedded_source = tf.nn.embedding_lookup(entity_embeddings, batch_source)
        embedded_r = tf.nn.embedding_lookup(relation_embeddings, batch_r)
        # embedded_target = tf.nn.embedding_lookup(entity_embeddings, batch_target)

        translated = embedded_source + embedded_r

        tiled_entity_embeddings = tf.tile(tf.expand_dims(entity_embeddings, axis=0), [batch_h.shape[0], 1, 1])
        tiled_translated = tf.tile(tf.expand_dims(translated, axis=1), [1, entity_embeddings.shape[0], 1])

        dis = compute_distance(tiled_translated, tiled_entity_embeddings)

        ranks = tf.argsort(tf.argsort(dis, axis=1), axis=1).numpy()
        target_ranks = ranks[np.arange(len(batch_target)), batch_target.numpy()]
        mean_ranks.extend(target_ranks)
    if epoch % 10 == 0:
        print("epoch = {}\tmean_rank = {}".format(epoch, np.mean(mean_ranks)))








