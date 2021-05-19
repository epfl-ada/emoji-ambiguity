import os
from itertools import chain

import numpy as np
import pandas as pd
from emoji import get_emoji_regexp
from p_tqdm import p_map

flatten = lambda x: list(chain.from_iterable(x))


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def save_to_csv(df, output_path):
    """
    Save pandas dataframe in a given path as a .csv without index column
    :param df: pandas dataframe
    :param output_path: absolute path where a file will be saved
    """
    df.to_csv(output_path + ".gz", encoding="utf-8", compression="gzip", index=False)


def apply_contains_emoji(df):
    return df.apply(contains_emoji)


def apply_find_emojis(df, text_col='tweet'):
    return df[text_col].apply(find_emojis)


def contains_emoji(text):
    return bool(get_emoji_regexp().search(text))


def find_emojis(text):
    """
    Output list of all emojis in a string
    """
    return get_emoji_regexp().findall(text)


def parallelize_dataframe(df, func, disable_progress_bar=False, n_cores=4):
    """
    :param disable_progress_bar: if set to True it will suppress tqdm output
    """
    df_split = np.array_split(df, n_cores)
    df = pd.concat(p_map(func, df_split, disable=disable_progress_bar, num_cpus=n_cores))
    return df
