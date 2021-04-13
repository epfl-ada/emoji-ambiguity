import argparse
import base64
import os

import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm


def _write_emoji_image(dir_path, emoji, image):
    with open(os.path.join(dir_path, f"{emoji}.png"),
              "wb") as fh:
        fh.write(base64.decodebytes(image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scrape emoji images from emojipedia to use for plotting')
    parser.add_argument('--output', action='store', required=True,
                        help="Directory's path where to dump emoji images")
    args = parser.parse_args()

    print("Scraping emojipedia...")
    http = urllib3.PoolManager()
    r = http.request('GET', "https://unicode.org/emoji/charts/full-emoji-list.html")
    soup = BeautifulSoup(r.data, features="html.parser")
    table = soup.find('table')

    tr = table.find('tr')
    td = tr.find("td", class_='andr')

    print("Parsing output and saving emojis...")
    pbar = tqdm(total=2000)
    while tr.findNext('tr'):
        pbar.update(1)
        try:
            td = tr.find("td", class_='andr')
            if td is not None and len(td['class']) == 1:
                if len(td.contents) > 0:
                    try:
                        image = td.contents[0]['src'].split(',')[1].encode("utf-8")
                        emoji = td.contents[0]['alt']
                        _write_emoji_image(args.output, emoji, image)
                    except TypeError:
                        image = td.contents[1]['src'].split(',')[1].encode("utf-8")
                        emoji = td.contents[1]['alt']
                        _write_emoji_image(args.output, emoji, image)
            tr = tr.findNext('tr')
        except AttributeError:
            tr = tr.findNext('tr')
            print("Couldn't find the table tag")
    pbar.close()
