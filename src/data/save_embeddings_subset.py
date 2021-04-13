# ignore futre numpy warnings in tensorflow used by spacy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import spacy

import argparse
import pickle
from collections import Counter

import gensim.downloader as api
import pandas as pd

from src.data.cldr import cldr_anns_to_df
from src.analysis.embedded import find_embedding
from src.data.utils import is_valid_file

# example usage
# python3 save_embeddings_subset.py
# --ambiguity-dataset local-data/ambiguity_dataset.csv.gz
# --cldr-description local-data/cldr_39_alpha4_en.xml
# --output embeddings_subset.pkl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Save subset of glove-twitter-200 gensim embeddings relevant to WUS and emojivec em dataset')
    parser.add_argument('--ambiguity-dataset', action='store', required=True,
                        type=lambda x: is_valid_file(parser, x),
                        help='Location of the ambiguity dataset')
    parser.add_argument('--cldr-description', action='store', required=True,
                        type=lambda x: is_valid_file(parser, x),
                        help='Location of CLDR emoji keyword English annotations')
    parser.add_argument('--output', action='store', required=True,
                        help='Location of the output pickle file')
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

    missing = 0
    embeddings_subset = {}
    tokenizer = spacy.load("en_core_web_sm")
    for emoji_description in combined_vocab:
        tokens = {token.text for token in tokenizer(emoji_description)}
        vec = find_embedding(tokens, word_embeddings)
        if vec is not None:
            embeddings_subset[emoji_description] = vec
        else:
            missing += 1
    print(f"Percentage of missing embeddings: {round(missing / len(combined_vocab) * 100)}%")

    with open(args.output, 'wb') as f:
        pickle.dump(embeddings_subset, f)
