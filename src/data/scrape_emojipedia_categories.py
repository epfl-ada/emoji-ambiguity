import argparse
import pickle

import urllib3
from bs4 import BeautifulSoup

from settings import EMOJI_CATEGORIZED

EMOJI_CATEGORIES = ["people", "nature", "food-drink", "activity",
                    "travel-places", "objects", "symbols", "flags"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scrape emoji categories from emojipedia')
    parser.add_argument('--output', default=EMOJI_CATEGORIZED,
                        help='Location of the output pickle file')
    args = parser.parse_args()

    categorized = {}
    http = urllib3.PoolManager()
    print("Scraping emojipedia...")
    for category in EMOJI_CATEGORIES:
        r = http.request('GET', f"https://emojipedia.org/{category}/")
        soup = BeautifulSoup(r.data, features='html.parser')
        categorized[category] = [el.text for el in soup.findAll('span', {'class': 'emoji'})][9:][:-74]

    # manual fixes
    categorized['people'] += ['\U0001f9b0', '\U0001f9b1', '\U0001f9b2', '\U0001f9b3']
    categorized['symbols'] += ['👁️‍🗨']

    print("Saving output...")
    with open(args.output, "wb") as f:
        pickle.dump(categorized, f)
