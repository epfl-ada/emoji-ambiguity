# ignore futre numpy warnings in tensorflow used by spacy
import warnings

from settings import AMBIGUITY_PATH, CLDR_ANNS_PATH

warnings.simplefilter(action='ignore', category=FutureWarning)

import spacy

import argparse
from collections import Counter

import gensim.downloader as api
import gensim.models as gs
import pandas as pd

from cldr import cldr_anns_to_df
from utils import is_valid_file
from src.analysis.embedded import find_embedding

# example usage
# python3 save_embeddings_subset.py
# --ambiguity-dataset local-data/ambiguity_dataset.csv.gz
# --cldr-description local-data/cldr_39_alpha4_en.xml
# --output glove-twitter-200-ambiguity.bin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Save subset of glove-twitter-200 gensim embeddings relevant to WUS and emojivec em dataset')
    parser.add_argument('--ambiguity-dataset', action='store', default=AMBIGUITY_PATH,
                        type=lambda x: is_valid_file(parser, x),
                        help='Location of the ambiguity dataset,'
                             ' default AMBIGUITY_PATH from settings.py')
    parser.add_argument('--cldr-description', action='store', default=CLDR_ANNS_PATH,
                        type=lambda x: is_valid_file(parser, x),
                        help='Location of CLDR emoji keyword English annotations, '
                             'default CLDR_ANNS_PATH from settings.py')
    parser.add_argument('--output', action='store', required=True,
                        help='Location of the output gensim KeyedVectors file')
    args = parser.parse_args()

    print("Reading original embeddings... (this may take a while)")
    word_embeddings = api.load('glove-twitter-200')

    cldr_anns = cldr_anns_to_df(args.cldr_description)
    emojis = pd.read_csv(args.ambiguity_dataset, encoding='utf-8')

    print("Choosing subset of words...")
    cldr_anns = cldr_anns.set_index("emoji")
    vocabularies = emojis[["emoji", "word"]] \
        .groupby("emoji").word \
        .apply(list).apply(Counter) \
        .reset_index() \
        .rename({"word": "vocabulary"}, axis=1).set_index("emoji")

    ambiguity_vocab = {st for row in vocabularies.vocabulary for st in row}

    vocabularies = vocabularies.join(cldr_anns)
    vocabularies = vocabularies[~vocabularies.cldr_description.isna()]
    cldr_vocab = {st for row in vocabularies.cldr_description for st in row}

    combined_vocab = ambiguity_vocab.union(cldr_vocab)

    print("Computing embeddings...")
    missing = 0
    tokenizer = spacy.load("en_core_web_sm")
    vector_size = word_embeddings.vector_size
    embeddings_subset = gs.KeyedVectors(word_embeddings.vector_size)
    for emoji_description in combined_vocab:
        tokens = {token.text for token in tokenizer(emoji_description)}
        vec = find_embedding(tokens, word_embeddings)
        if vec is not None:
            # we can't decode space later
            description = emoji_description.replace(" ", "-")
            embeddings_subset.add(description, vec)
        else:
            missing += 1
    print(f"Percentage of missing embeddings: {round(missing / len(combined_vocab) * 100)}%")

    # embeddings_subset.save_word2vec_format(args.output, binary=True)
    embeddings_subset.save_word2vec_format(args.output, binary=True)
