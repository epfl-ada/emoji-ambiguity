import argparse

import numpy as np
import pandas as pd

from src.data.utils import parallelize_dataframe, save_to_csv, apply_find_emojis

pd.set_option('mode.chained_assignment', None)


# example usage
# python src/data/sample_tweets.py --input /scratch/czestoch/emojitweets-01-04-2018.txt.gz
# --output /scratch/czestoch/sampled_tweets.txt --N 30 --num-cpus 24

def sample_tweets_by_emojis(tweets, sample_size=30, num_cpus=24):
    tweets['emojis'] = parallelize_dataframe(tweets, apply_find_emojis, n_cores=num_cpus)
    gb = tweets.explode('emojis').groupby("emojis")
    emojis_idx_mapping = gb.groups
    chosen_indices = np.array([])
    for _, indices in emojis_idx_mapping.items():
        try:
            chosen = np.random.choice(indices.values, size=sample_size)
        except ValueError:
            chosen = indices
        chosen_indices = np.append(chosen_indices, chosen)
    return tweets.iloc[chosen_indices]


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
    print("Reading data in...")
    tweets = pd.read_table(args.input, header=None, lineterminator='\n', encoding='utf-8')
    tweets = tweets.rename({0: "tweet"}, axis=1)
    print("Sampling...")
    out = sample_tweets_by_emojis(tweets, sample_size=args.N, num_cpus=args.num_cpus)
    print("Saving...")
    save_to_csv(out, args.output)
