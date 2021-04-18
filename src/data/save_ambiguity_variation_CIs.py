import argparse
from collections import Counter

import pandas as pd

from analysis.embedded import calculate_vocabulary_variation, embedded_CIs
from analysis.embedded import read_embeddings
from settings import AMBIGUITY_PATH, EMBEDDINGS_PATH
from utils import parallelize_dataframe, save_to_csv

pd.set_option('mode.chained_assignment', None)

# example usage
# python3 save_ambiguity_variation_CIs.py
# --output data/interim/TESTambiguity_variation.csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate bootstrapped CIs '
                                                 'for emoji semantic variation in the embedded space and save them')
    parser.add_argument('--output', action='store', required=True,
                        help='Location of the output csv with emoji'
                             ' semantic variation with confidence intervals')
    parser.add_argument('--num-cpus', type=int,
                        help='Number of cores to use for computing confidence intervals')
    args = parser.parse_args()

    print("Reading data...")
    emojis = pd.read_csv(AMBIGUITY_PATH, encoding='utf-8')
    word_embeddings = read_embeddings(EMBEDDINGS_PATH)

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
    CIs = parallelize_dataframe(vocabularies, func, n_cores=args.num_cpus)
    vocabularies["CIs"] = CIs

    print("Saving csv...")
    save_to_csv(vocabularies, args.output)
