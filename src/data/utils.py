import os
from itertools import chain

flatten = lambda x: list(chain.from_iterable(x))


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def save_to_csv(df, output_path, sep=","):
    """
    Save pandas dataframe in a given path as a .csv without index column
    :param df: pandas dataframe
    :param output_path: absolute path where a file will be saved
    :param sep: separator to use to save the file
    """
    df.to_csv(output_path + ".gz", encoding="utf-8", compression="gzip", index=False, sep=sep)
