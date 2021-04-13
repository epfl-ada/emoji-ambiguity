import argparse
import pickle
from collections import Counter

import pandas as pd

from settings import EMBEDDINGS_PATH, AMBIGUITY_PATH, CLDR_ANNS_PATH
from src.analysis.embedded import calculate_cldr_distance, embedded_CIs
from src.data.cldr import cldr_anns_to_df
from src.data.utils import flatten, parallelize_dataframe, save_to_csv

pd.set_option('mode.chained_assignment', None)

# example usage
# python3 save_cldr_distance_CIs.py
# --output data/interim/cldr_distance.csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate bootstrapped CIs '
                                                 'for emoji distance to its CLDR description and save them')
    parser.add_argument('--output', action='store', required=True,
                        help='Location of the output csv with emoji'
                             ' vocabularies distances to CLDR descriptions with confidence intervals')
    args = parser.parse_args()

    print("Reading data...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        word_embeddings = pickle.load(f)
    emojis = pd.read_csv(AMBIGUITY_PATH, encoding='utf-8')

    print("Preprocessing...")
    cldr_anns = cldr_anns_to_df(CLDR_ANNS_PATH)
    cldr_anns = cldr_anns.set_index("emoji")

    vocabularies = emojis[["emoji", "word"]] \
        .groupby("emoji") \
        .word.apply(list).apply(Counter) \
        .rename({"word": "vocabulary"}, axis=1) \
        .to_frame()

    vocabularies = vocabularies.join(cldr_anns)
    vocabularies.cldr_description.isna().sum() / len(vocabularies)

    print("Calculating distances...")
    # no use when there is no description
    vocabularies = vocabularies[~vocabularies.cldr_description.isna()]
    vocabularies["words"] = vocabularies.word.apply(lambda x: set(flatten(map(str.split, list(x.keys())))))
    vocabularies = vocabularies.rename({"word": "vocabulary"}, axis=1)
    vocabularies.cldr_description = vocabularies.cldr_description.apply(lambda x: set(flatten(map(str.split, x))))
    vocabularies["cldr_distance"] = vocabularies.apply(lambda row: calculate_cldr_distance(row.vocabulary,
                                                                                           word_embeddings,
                                                                                           row.cldr_description),
                                                       axis=1)

    print("Bootstrapping...")
    func = lambda partial_df: partial_df.apply(lambda row: embedded_CIs(calculate_cldr_distance,
                                                                        row.vocabulary,
                                                                        word_embeddings,
                                                                        cldr_desc=row.cldr_description), axis=1)
    CIs = parallelize_dataframe(vocabularies, func)
    vocabularies["CIs"] = CIs

    print("Saving csv...")
    save_to_csv(vocabularies, args.output)
