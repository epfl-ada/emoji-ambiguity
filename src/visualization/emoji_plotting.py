import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from settings import EMOJI_IMGS


def get_emoji(emoji, log):
    try:
        directory_path = EMOJI_IMGS
        if emoji == '🛰️':
            path = os.path.join(directory_path, "satellite.png")
            return plt.imread(path)
        elif emoji == '◼️':
            path = os.path.join(directory_path, "medium-black-square.png")
            return plt.imread(path)
        elif emoji == '✈️':
            path = os.path.join(directory_path, "airplane.png")
            return plt.imread(path)
        elif emoji == '⬆️':
            path = os.path.join(directory_path, "arrow_up.png")
            return plt.imread(path)
        path = os.path.join(directory_path, f"{emoji}.png")
        return plt.imread(path)
    except FileNotFoundError:
        if log:
            print(f"{path} not found")
        path = os.path.join(directory_path, "◻.png")
        return plt.imread(path)


def offset_image(coord, name, ax, log):
    img = get_emoji(name, log=log)
    if name in ["⛩", "🏚", "☄", "☪", "🎛", "🎚", "🖲", "↕"]:
        im = OffsetImage(img, zoom=0.01)
    else:
        im = OffsetImage(img, zoom=0.3)
    im.image.axes = ax
    ab = AnnotationBbox(im, (coord, 0), xybox=(0., -16.), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)


def emoji_scatter(x, y, emoji, ax=None, zoom=1, log=False):
    if ax is None:
        ax = plt.gca()
    image = get_emoji(emoji, log=log)
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def get_emojis_per_bin(ax, vocabs):
    df = {"bin": [], "example_emojis": []}
    bins = get_hist(ax)
    bin_low_limit = bins[0]
    for bin_up_limit in bins[1:]:
        emojis = vocabs[(vocabs.entropy > bin_low_limit) & (vocabs.entropy < bin_up_limit)].index.values
        try:
            bin_emojis = random.sample(list(emojis), 3)
        except ValueError:
            bin_emojis = list(emojis)
        df["bin"].append((round(bin_low_limit, 2), round(bin_up_limit, 2)))
        df["example_emojis"].append(bin_emojis)
        bin_low_limit = bin_up_limit
    return pd.DataFrame(df)


def get_hist(ax):
    bins = []
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        bins.append(x0)  # left edge of each bin
    bins.append(x1)  # also get right edge of last bin
    return bins
