# coding=utf-8
import numpy as np


def hits_score(ranks, k):
    if not isinstance(ranks, np.ndarray):
        ranks = np.array(ranks)
    num_hits = len(np.where(ranks < k)[0])
    hits_score = num_hits / len(ranks)
    return hits_score