import os
import zipfile
import json
import time
from datetime import datetime
import pytz
from multiprocessing import Pool
import preprocessor as tweet_preprocessor
import pandas as pd
import argparse


def process_emoji(file_paths, emoji_dir, save_path, num_cpus):
    p = Pool(num_cpus)
    print('Parallelized on number of cores:', num_cpus)
    output = p.map(process_file, file_paths)
    p.close()
    p.join()
    # output = process_file("tweets_dump.json")
    pd.DataFrame([item for sublist in output for item in sublist])\
        .to_parquet(os.path.join(save_path, f"{os.path.splitext(emoji_dir)[0]}.parquet"),\
                    compression='gzip')


def process_file(archive_path):
    out = []
    try:
        if os.path.isfile(archive_path):
            archive = zipfile.ZipFile(archive_path, 'r')
            filename = os.path.splitext(os.path.basename(archive_path))[0]
            with archive.open(filename, "r") as fp:
                for line in fp:
                    try:
                        tweet_json = json.loads(line)
                        if 'retweeted_status' in tweet_json:
                            continue
                        else:
                            try:
                                entry = {}
                                referenced_tweets = tweet_json.get("referenced_tweets", [{"type": "no_referenced_tweets"}])
                                if referenced_tweets[0]['type'] != 'retweeted' and 'RT' not in tweet_json['text']:
                                    text = tweet_json['text']
                                    text = text.replace('&amp;','&')
                                    text = text.replace('&amp','&')
                                    text = text.replace('&gt;','&')
                                    text = text.replace('&gt','&')
                                    text = text.replace('&lt;','&')
                                    text = text.replace('&lt','&')
                                    entry['n_chars'] = len(text)
                                    entry['tweet_id'] = tweet_json['id']
                                    entry['created_at'] = datetime.strptime(tweet_json['created_at'], "%Y-%m-%dT%H:%M:%S.000Z").replace(tzinfo=pytz.UTC)
                                    entry['lang'] = tweet_json['lang']
                                    entry['user_id'] = tweet_json['author']['id']
                                    entry['user_n_followers'] = tweet_json['author']['public_metrics']['followers_count']
                                    entry['user_n_following'] = tweet_json['author']['public_metrics']['following_count']
                                    entry['user_n_tweets'] = tweet_json['author']['public_metrics']['tweet_count']
                                    entry['text'] = tweet_preprocessor.clean(text)
                                    out.append(entry)
                            except (KeyError) as e:
                                print(e)
                                pass
                    except (json.decoder.JSONDecodeError) as e:
                        print(e)
                        continue
            return out
        else:
            return out
    except (OSError, EOFError) as e:
        print(e)
        return out


if __name__ == "__main__":

    # output = process_file("/scratch/czestoch/twitter-stream/keycap_#/keycap_#_start2019-09-01_stop2019-10-31_0.json.zip")
    # pd.DataFrame(output)\
    #     .to_parquet("output.parquet")

    parser = argparse.ArgumentParser(
        # TODO: change twitter-stream to twitter-api, change dir name
        description="Process tweets from twitter seatch API compressed .zip jsons to dataframes and save them as parquet files, clean tweets from hashtags, mentions and urls")
    parser.add_argument('--input', default="/scratch/czestoch/twitter-api-emojis",help="Path to the directory with emoji directories containing zipped json files with batches of tweets to be processed")
    parser.add_argument('--output', required=True, help="Path to the output parquet dataframes with english tweets containing emojis")
    parser.add_argument('--num-cpus', required=False, default=30, help="Number of CPU cores to use with multiprocessing, default: 30")
    args = parser.parse_args()

    tweet_preprocessor.set_options(tweet_preprocessor.OPT.URL,\
       tweet_preprocessor.OPT.MENTION, tweet_preprocessor.OPT.HASHTAG)
       
    for emoji_dir in os.listdir(args.input):
        print(f"Now processing emoji {os.path.basename(emoji_dir)}")
        file_paths = [os.path.join(args.input, emoji_dir, filename) for filename in os.listdir(os.path.join(args.input, emoji_dir))]
        start = time.time()
        process_emoji(file_paths, emoji_dir, args.output, args.num_cpus)
        end = time.time()
        elapsed = end - start
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
