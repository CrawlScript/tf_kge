# coding=utf-8

from tf_kge.dataset.wn18 import WN18Dataset

train_kg, test_kg, valid_kg, entity_indexer, relation_indexer = WN18Dataset().load_data()

print(train_kg.head_relation_tail_dict)

# print(train_heads.shape, test_heads.shape, valid_heads.shape)
