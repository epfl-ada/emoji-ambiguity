import pickle

from settings import EMOJI_CATEGORIZED
from src.data.emoji_categorization import fine_grained_categories


def assign_emojipedia_category(emoji):
    with open(EMOJI_CATEGORIZED, 'rb') as f:
        categorized = pickle.load(f)
    for category in categorized:
        if emoji in categorized[category]:
            return category


def assign_our_category(row):
    for subcategory in fine_grained_categories:
        if row.emoji in fine_grained_categories[subcategory]:
            return subcategory
    return row.category
