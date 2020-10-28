# coding=utf-8

from tqdm import tqdm

class KG(object):
    def __init__(self, heads, relations, tails,
                 entity_indexer, relation_indexer):
        self.heads = heads
        self.relations = relations
        self.tails = tails

        self.entity_indexer = entity_indexer
        self.relation_indexer = relation_indexer

        self.head_relation_tail_dict = self.build_triple_dict(heads, relations, tails)
        self.tail_relation_head_dict = self.build_triple_dict(tails, relations, heads)

    def build_triple_dict(self, source_entities, relations, target):
        source_relation_target_dict = {}

        for source, relation, target in tqdm(zip(source_entities, relations, target)):
            if source not in source_relation_target_dict:
                relation_target_dict = {}
                source_relation_target_dict[source] = relation_target_dict
            else:
                relation_target_dict = source_relation_target_dict[source]

            if relation not in relation_target_dict:
                targets = set()
                relation_target_dict[relation] = targets
            else:
                targets = relation_target_dict[relation]

            targets.add(target)
        return source_relation_target_dict



    @property
    def num_entities(self):
        return len(self.entity_indexer)

    @property
    def num_relations(self):
        return len(self.relation_indexer)

    @property
    def num_triples(self):
        return len(self.heads)

    def __str__(self):
        return "KG: entities => {}\trelations => {} triples => {}".format(self.num_entities, self.num_relations, self.num_triples)
