# coding=utf-8

import numpy as np
import os
from tqdm import tqdm
import numpy as np

from tf_geometric.data.dataset import DownloadableDataset
import json

class CommonDataset(DownloadableDataset):

    def _read_id_index_info(self, id_index_path):
        print("reading id_index_info: ", id_index_path)
        id_index_dict = {}
        index_id_dict = {}
        with open(id_index_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                id = items[0]
                index = int(items[1])

                id_index_dict[id] = index
                index_id_dict[index] = id
        return id_index_dict, index_id_dict

    def _read_triples(self, triple_path, entity_id_index_dict, relation_id_index_dict):
        triples = []
        print("reading triples: ", triple_path)
        with open(triple_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                head_entity_id, tail_entity_id, relation_id = line.split()
                triple = [
                    entity_id_index_dict[head_entity_id],
                    relation_id_index_dict[relation_id],
                    entity_id_index_dict[tail_entity_id]
                ]
                triples.append(triple)
        triples = np.array(triples)
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        return heads, relations, tails

    def process(self):

        data_dir = os.path.join(self.raw_root_path, self.dataset_name)

        test_triple_path = os.path.join(data_dir, "test.txt")
        train_triple_path = os.path.join(data_dir, "train.txt")
        valid_triple_path = os.path.join(data_dir, "valid.txt")

        entity_path = os.path.join(data_dir, "entity2id.txt")
        relation_path = os.path.join(data_dir, "relation2id.txt")


        entity_id_index_dict, entity_index_id_dict = self._read_id_index_info(entity_path)
        relation_id_index_dict, relation_index_id_dict = self._read_id_index_info(relation_path)

        train_heads, train_relations, train_tails = self._read_triples(train_triple_path, entity_id_index_dict, relation_id_index_dict)
        test_heads, test_relations, test_tails = self._read_triples(test_triple_path, entity_id_index_dict, relation_id_index_dict)
        valid_heads, valid_relations, valid_tails = self._read_triples(valid_triple_path, entity_id_index_dict, relation_id_index_dict)

        return (entity_id_index_dict, entity_index_id_dict), \
               (relation_id_index_dict, relation_index_id_dict), \
               (train_heads, train_relations, train_tails), \
               (test_heads, test_relations, test_tails), \
               (valid_heads, valid_relations, valid_tails)
