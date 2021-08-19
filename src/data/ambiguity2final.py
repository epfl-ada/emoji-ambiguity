import argparse
import pickle

import pandas as pd

from settings import EMOJI_CATEGORIZED, AMBIGUITY_PATH, TWITTER_COUNTS, FINAL_DF
from emoji_categorization import fine_grained_categories
from utils import is_valid_file, save_to_csv


def assign_emojipedia_category(emoji):
    for category in categorized:
        if emoji in categorized[category]:
            return category


def assign_our_category(row):
    for subcategory in fine_grained_categories:
        if row.emoji in fine_grained_categories[subcategory]:
            return subcategory
    return row.category


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate final ambiguity dataset with fine-grained emoji categories and twitter counts')
    parser.add_argument('--ambiguity', default=AMBIGUITY_PATH,
                        type=lambda x: is_valid_file(parser, x),
                        help="Location of the ambiguity dataset: 'ambiguity_dataset.csv.gz'")
    parser.add_argument('--twitter-count', default=TWITTER_COUNTS,
                        type=lambda x: is_valid_file(parser, x),
                        help="Location of the twitter counts: 'twitter-api-context-free-emoji-counts.csv.gz'")
    parser.add_argument('--output', default=FINAL_DF,
                        help='Location of the final dataset')
    args = parser.parse_args()

    with open(EMOJI_CATEGORIZED, "rb") as f:
        categorized = pickle.load(f)

    print("Assigning categories...")
    ambiguity = pd.read_csv(args.ambiguity, encoding='utf-8')
    ambiguity["category"] = ambiguity.emoji.apply(assign_emojipedia_category)
    ambiguity = ambiguity[ambiguity.category != "flags"]

    ambiguity["category"] = ambiguity.apply(assign_our_category, axis=1)
    ambiguity = ambiguity.replace({"symbols": "symbols & signs"})

    print("Assigning twitter counts...")
    twitter_count = pd.read_csv(args.twitter_count)
    ambiguity = pd.merge(ambiguity, twitter_count, on='emoji', how='left').rename({"count": "twitter_count"}, axis=1)
    ambiguity = ambiguity.fillna(0)

    print("Saving to csv...")
    save_to_csv(ambiguity, args.output)
