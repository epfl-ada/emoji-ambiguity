import argparse
from collections import Counter

import pandas as pd

from src.analysis.variation import calculate_vocabulary_variation, embedded_CIs
from src.analysis.variation import read_embeddings
from settings import AMBIGUITY_PATH, EMBEDDINGS_PATH, AMBIGUITY_CLUSTER, EMBEDDINGS_CLUSTER
from src.data.utils import parallelize_dataframe, save_to_csv

pd.set_option('mode.chained_assignment', None)

# example usage
# python3 save_ambiguity_variation_CIs.py
# --output data/interim/ambiguity_variation.csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate bootstrapped CIs '
                                                 'for emoji semantic variation in the embedded space and save them')
    parser.add_argument('--cluster', action='store_true', default=False,
                        help='When cluster is True, use cluster path to the input dataset and embeddings')
    parser.add_argument('--output', required=True,
                        help='Location of the output csv with emoji'
                             ' semantic variation with confidence intervals')
    parser.add_argument('--num-cpus', type=int,
                        help='Number of cores to use for computing confidence intervals')
    args = parser.parse_args()

    print("Reading data...")
    data_path = AMBIGUITY_PATH
    embeddings_path = EMBEDDINGS_PATH
    if args.cluster:
        data_path = AMBIGUITY_CLUSTER
        embeddings_path = EMBEDDINGS_CLUSTER
    emojis = pd.read_csv(data_path, encoding='utf-8')
    word_embeddings = read_embeddings(embeddings_path)

    print("Preprocessing...")
    vocabularies = emojis[["emoji", "word"]] \
        .groupby("emoji").word \
        .apply(list).apply(Counter) \
        .reset_index() \
        .rename({"word": "vocabulary"}, axis=1)
    vocabularies.head()

    print("Calculating semantic variation in embedding space per emoji...")
    vocabularies[["variation", "mode_embedding"]] = vocabularies.apply(lambda row:
                                                                       calculate_vocabulary_variation(row.vocabulary,
                                                                                                      word_embeddings),
                                                                       axis=1,
                                                                       result_type='expand')

    print("Bootstrapping...")
    func = lambda partial_df: partial_df.apply(lambda row: embedded_CIs(calculate_vocabulary_variation,
                                                                        row.vocabulary, row.mode_embedding,
                                                                        word_embeddings),
                                               axis=1)
    CIs = parallelize_dataframe(vocabularies, func, n_cores=args.num_cpus)
    vocabularies["CIs"] = CIs

    print("Saving csv...")
    save_to_csv(vocabularies, args.output)
