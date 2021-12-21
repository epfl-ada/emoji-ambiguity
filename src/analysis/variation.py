import random
from collections import Counter
from functools import partial

import numpy as np
from gensim import models as gs
from scipy.spatial.distance import cosine as cosine_distance

from src.data.utils import flatten


def calculate_vocabulary_variation(element, embeddings, mode_embedding=None):
    if mode_embedding is None:
        mode_embedding = get_mode_embedding(element, embeddings)
    if mode_embedding is not None:
        total_count = sum(element.values())
        mode_vec = np.expand_dims(mode_embedding, 0)
        emocab_variance = 0
        for emoji, count in element.items():
            emoji_vec = get_embedding(embeddings, emoji)
            if emoji_vec is not None:
                emoji_vec = np.expand_dims(emoji_vec, 0)
                distance = cosine_distance(mode_vec, emoji_vec)
                emocab_variance += (count / total_count) * distance
        return emocab_variance, mode_embedding
    return np.nan, mode_embedding


def get_mode_embedding(element, embeddings):
    mode_emoji = element.most_common(1)[0][0]
    return get_embedding(embeddings, mode_emoji)


def get_embedding(embeddings, word):
    try:
        return embeddings.get_vector(word)
    except KeyError:
        return None


def find_embedding(description, embeddings):
    vec = None
    # if description is one word and embedding is in vocabulary simply use it
    if len(description) == 1:
        word = description.pop()
        if word in embeddings.vocab:
            vec = embeddings[word]
    # join the words, sometimes embeddings appear in splitted
    # or joined form in the vocabulary,
    # eg. there may be an embedding for 'toilet paper' or 'toiletpaper'
    elif "".join(description) in embeddings.vocab:
        vec = embeddings["".join(description)]
    # if description is longer then take average of embeddings
    elif len(description) > 1:
        if all([word in embeddings.vocab for word in description]):
            vec = np.array([embeddings[word] for word in description]).sum(axis=0)
            vec /= len(description)
        else:
            return None
    return vec


def read_embeddings(embeddings_path):
    word_embeddings = gs.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    word_embeddings.vocab = {description.replace("-", " "): vector for description, vector in
                             word_embeddings.vocab.items()}
    return word_embeddings


def embedded_CIs(func, vocabulary, mode_embedding, word_embeddings, num_draws=2, alpha=5, **kwargs):
    annotations = flatten([[k] * v for k, v in vocabulary.items()])
    sampler = partial(resampling, func=func, annotations=annotations, word_embeddings=word_embeddings,
                      mode_embedding=mode_embedding, **kwargs)
    results = list(zip(*map(sampler, range(num_draws))))[0]
    results = np.array(results)
    return [np.nanpercentile(results, alpha / 2), np.nanpercentile(results, 100 - (alpha / 2))]


def resampling(num_draw, func, annotations, word_embeddings, **kwargs):
    sampled = Counter(random.choices(annotations, k=len(annotations)))
    return func(sampled, word_embeddings, **kwargs)
