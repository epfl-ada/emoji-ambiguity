import argparse

import numpy as np
import pandas as pd
from emoji import get_emoji_regexp

from src.data.utils import parallelize_dataframe, save_to_csv, apply_find_emojis

pd.set_option('mode.chained_assignment', None)


# example usage
# python src/data/sample_tweets.py --input /scratch/czestoch/emojitweets-01-04-2018.txt.gz
# --output /scratch/czestoch/sampled_tweets.txt --N 30 --num-cpus 24

def sample_tweets_by_emojis(tweets, sample_size, num_cpus):
    tweets['emojis'] = parallelize_dataframe(tweets, apply_find_emojis, n_cores=num_cpus)
    # First remove tweets that differ only by emoji, some common advertising tweets
    tweets['masked_tweets'] = parallelize_dataframe(tweets.tweet, replace_emojis, n_cores=num_cpus)
    tweets = tweets.drop_duplicates("masked_tweets")[["emojis", "tweet"]]
    # Explode emojis, we will use a tweet only once even if it contains multiple different emojis
    tweets = tweets.explode("emojis").drop_duplicates("tweet")
    emojis_idx_mapping = tweets.groupby("emojis").groups
    chosen_indices = np.array([])
    for _, indices in emojis_idx_mapping.items():
        try:
            chosen = np.random.choice(indices.values, size=sample_size, replace=False)
        except ValueError:
            chosen = []
        chosen_indices = np.append(chosen_indices, chosen)
    return tweets.loc[chosen_indices]

def replace_emojis(tweet):
    return tweet.replace(emoji_regex, "[EMOJI]", regex=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Downsample kaggle emoji tweets dataset to get N tweets per emoji')
    parser.add_argument('--input', required=True,
                        help="Path to the input emoji tweet dataset from kaggle: emojitweets-01-04-2018.txt.gz")
    parser.add_argument('--output', required=True, help="Path to the output downsampled .txt")
    parser.add_argument('--N', type=int, default=30, help="How many tweets to sample per emoji")
    parser.add_argument('--num-cpus', type=int, default=4, help="How many cores to use for processing")
    args = parser.parse_args()

    np.random.seed(42)
    emoji_regex = get_emoji_regexp()
    print("Reading data in...")
    tweets = pd.read_table(args.input, header=None, lineterminator='\n', quoting=3, encoding='utf-8')
    tweets = tweets.rename({0: "tweet"}, axis=1)
    tweets = tweets.drop_duplicates('tweet')
    tweets = tweets.reset_index()
    print("Sampling...")
    out = sample_tweets_by_emojis(tweets, sample_size=args.N, num_cpus=args.num_cpus)[["tweet", "emojis"]]
    print("Saving...")
    save_to_csv(out, args.output)
