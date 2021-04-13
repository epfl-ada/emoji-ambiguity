import argparse
import pickle
from collections import Counter

import pandas as pd

from settings import AMBIGUITY_PATH, EMBEDDINGS_PATH
from src.analysis.embedded import calculate_vocabulary_variation, embedded_CIs
from src.data.utils import parallelize_dataframe
from src.data.utils import save_to_csv

pd.set_option('mode.chained_assignment', None)

# example usage
# python3 save_ambiguity_variation_CIs.py
# --output data/interim/ambiguity_variation.csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate bootstrapped CIs '
                                                 'for emoji semantic variation in the embedded space and save them')
    parser.add_argument('--output', action='store', required=True,
                        help='Location of the output csv with emoji'
                             ' semantic variation with confidence intervals')
    args = parser.parse_args()

    print("Reading data...")
    emojis = pd.read_csv(AMBIGUITY_PATH, encoding='utf-8')
    with open(EMBEDDINGS_PATH, "rb") as f:
        word_embeddings = pickle.load(f)

    print("Preprocessing...")
    vocabularies = emojis[["emoji", "word"]] \
        .groupby("emoji").word \
        .apply(list).apply(Counter) \
        .reset_index() \
        .rename({"word": "vocabulary"}, axis=1)
    vocabularies.head()

    print("Calculating semantic variation in embedding space per emoji...")
    vocabularies["variation"] = vocabularies.apply(lambda row:
                                                   calculate_vocabulary_variation(row.vocabulary,
                                                                                  word_embeddings), axis=1)

    print("Bootstrapping...")
    func = lambda partial_df: partial_df.apply(lambda row: embedded_CIs(calculate_vocabulary_variation,
                                                                        row.vocabulary, word_embeddings),
                                               axis=1)
    CIs = parallelize_dataframe(vocabularies, func)
    vocabularies["CIs"] = CIs

    print("Saving csv...")
    save_to_csv(vocabularies, args.output)
