import os

import argparse
import pandas as pd
from pandarallel import pandarallel

from collections import Counter

from src.data.utils import find_emojis, save_to_csv
from src.data.utils import flatten
from settings import AMBIGUITY_VARIATION


def main(args):
    pandarallel.initialize(nb_workers=args.num_cpus)
    df = pd.concat([pd.read_parquet(os.path.join(args.input, filename)) for filename in os.listdir(args.input)])
    df = df.drop_duplicates() \
        .drop_duplicates("text")

    df['emojis'] = df.text.parallel_apply(find_emojis)
    df['n_emojis'] = df.emojis.parallel_apply(len)
    df = df[df.n_emojis >= 1]

    emoji_counts = Counter(flatten(df['emojis'].apply(set).tolist()))
    df2 = {"emoji": list(emoji_counts.keys()), "count": list(emoji_counts.values())}
    df2 = pd.DataFrame.from_records(df2)
    df2 = df2[df2.emoji.isin(pd.read_csv(AMBIGUITY_VARIATION).emoji.tolist())]

    save_to_csv(df2, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument('--input', default='/scratch/czestoch/twitter-api-emojis-parquets',
                        help="Path to the tweets with our emojis from twitter API from September and October 2019")
    parser.add_argument('--output',
                        default="/scratch/czestoch/emoji-measures/twitter-api-context-free-emoji-counts.csv",
                        help="Path to the output csv with emoji twitter counts")
    parser.add_argument('--num-cpus', default=30, help="Number of CPUs to use to extract emojis from tweets")
    args = parser.parse_args()

    main(args)
