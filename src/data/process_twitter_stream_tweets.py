import os
import bz2
import json
import time
from datetime import datetime
import pytz
from multiprocessing import Pool
import pandas as pd
import argparse

from src.data.utils import find_emojis
from src.data.prepare_tweet_paths import batchlist_path


def process_batch(file_paths, batch_filename, save_path, num_cpus):
    p = Pool(num_cpus)
    print('Parallelized on number of cores:', num_cpus)
    output = p.map(process_file, file_paths)
    p.close()
    p.join()
    pd.DataFrame([item for sublist in output for item in sublist])\
        .to_parquet(os.path.join(save_path, f"{os.path.splitext(batch_filename)[0]}.parquet"),\
                    compression='gzip')


def process_file(file):
    out = []
    try:
        if os.path.isfile(file):
            with bz2.open(file, "rt",encoding="utf-8") as fp:  
                for cnt, line in enumerate(fp):
                    try:
                        f = json.loads(line)
                        if 'retweeted_status' in f:
                            continue
                        else:
                            try:
                                if f['lang'] == 'en':
                                    entry = {}
                                    #this happens if a tweet is deleted
                                    #if 'id' not in f.keys():
                                    #    continue
                                    if f['truncated'] == False:
                                        text = f['text']
                                    else:
                                        rang = f['extended_tweet']['display_text_range']
                                        text = f['extended_tweet']['full_text'][rang[0]:rang[1]]
                                    if 'htt' in text or not find_emojis(text):
                                        continue
                                    text = text.replace('&amp;','&')
                                    text = text.replace('&amp','&')
                                    text = text.replace('&gt;','&')
                                    text = text.replace('&gt','&')
                                    text = text.replace('&lt;','&')
                                    text = text.replace('&lt','&')
                                    entry['n_chars'] = len(text)
                                    entry['tweet_id'] = f['id_str']
                                    entry['created_at'] = datetime.strptime(f['created_at'],
                                                                            '%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.UTC)
                                    entry['lang'] = f['lang']
                                    entry['user_id'] = f['user']['id_str']
                                    entry['user_n_followers'] = f['user']['followers_count']
                                    entry['user_n_following'] = f['user']['friends_count']
                                    entry['user_n_tweets'] = f['user']['statuses_count']
                                    entry['text'] = text
                                    out.append(entry)
                            except (KeyError) as e:
                                pass
                    except (json.decoder.JSONDecodeError) as e:
                        continue
            return out
        else:
            return out
    except (OSError, EOFError) as e:
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process tweets from twitter stream compressed jsons to dataframes and save them as parquet files, filter English tweets with emojis")
    parser.add_argument('--input', help="Path to the directory with .txt files containing paths for batches of tweets to be processed"\
    "if not given batchlist_path variable from prepare_tweet_paths will be used")
    parser.add_argument('--output', required=True, help="Path to the output parquet dataframes with english tweets containing emojis")
    parser.add_argument('--num-cpus', required=False, default=30, help="Number of CPU cores to use with multiprocessing, default: 30")
    args = parser.parse_args()

    batchlist_filepaths = args.input
    if args.input is None:
        batchlist_filepaths = batchlist_path

    for batch_filename in os.listdir(batchlist_filepaths):
        print(f"Now processing batch file {batch_filename}")
        with open(os.path.join(batchlist_filepaths, batch_filename), "rt") as fp:
            file_paths = list(map(lambda x: x.rstrip('\n'), fp.readlines()))
        start = time.time()
        process_batch(file_paths, batch_filename, args.output, args.num_cpus)
        end = time.time()
        elapsed = end - start
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))    