# coding=utf-8

from tf_kge.dataset.wn18 import WN18Dataset

(entity_id_index_dict, entity_index_id_dict), (relation_id_index_dict, relation_index_id_dict), \
    (train_heads, train_relations, train_tails), (test_heads, test_relations, test_tails), \
    (valid_heads, valid_relations, valid_tails) = WN18Dataset().load_data()

print(test_tails)
