import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from settings import EMOJI_IMGS


def plot_emoji_barplot(df, ax, col):
    emoji_ticks = df.emoji.to_list()
    CIs = np.array(df.CIs.to_list()).T
    low = df[col].values - CIs[0, :]
    high = CIs[1, :] - df[col].values
    ax.bar(range(len(emoji_ticks)), df[col].to_list(), yerr=np.vstack((low, high)))
    for i, c in enumerate(emoji_ticks):
        offset_image(i, c, ax)
    sns.barplot(data=df, x=df.index, y=col, ax=ax)


def get_emoji(emoji, log=False):
    try:
        path = os.path.join(EMOJI_IMGS, f"{emoji}.png")
        return plt.imread(path)
    except FileNotFoundError:
        if log:
            print(f"{emoji} not found")
        path = os.path.join(EMOJI_IMGS, "◻.png")
        return plt.imread(path)


def offset_image(coord, name, ax):
    img = get_emoji(name)
    if name in ["⛩", "🏚", "☄", "☪", "🎛", "🎚", "🖲", "↕"]:
        im = OffsetImage(img, zoom=0.01)
    else:
        im = OffsetImage(img, zoom=0.3)
    im.image.axes = ax
    ab = AnnotationBbox(im, (coord, 0), xybox=(0., -16.), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)


def emoji_scatter(x, y, emoji, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    image = get_emoji(emoji)
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
