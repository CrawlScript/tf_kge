# coding=utf-8
from tf_kge.data.kg import KG
import numpy as np


def entity_negative_sampling(source_entities, relations, kg: KG, target_entity_type, filtered=False):
    """

    :param source_entities:
    :param relations:
    :param kg:
    :param filtered:
    :param target_entity_type:  "head" | "tail"
    :return:
    """

    if not filtered:
        return np.random.randint(0, kg.num_entities, len(source_entities))

    if target_entity_type == "head":
        source_relation_target_dict = kg.head_relation_tail_dict
    else:
        source_relation_target_dict = kg.tail_relation_head_dict

    negative_targets = []
    for source, relation in zip(source_entities, relations):
        positive_targets = source_relation_target_dict[source][relation]
        while True:
            random_target = np.random.randint(0, kg.num_entities)
            if random_target not in positive_targets:
                negative_targets.append(random_target)
                break
    return negative_targets


