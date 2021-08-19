import argparse
import datetime
import os
import zipfile

import emoji as em
import jsonlines
import pandas as pd
from requests import HTTPError
from twarc import Twarc2, expansions

from settings import TWITTER_TOKEN, AMBIGUITY_VARIATION

client = Twarc2(bearer_token=TWITTER_TOKEN)


def main(args):
    emojis = pd.read_csv(AMBIGUITY_VARIATION).emoji.values
    compression = zipfile.ZIP_DEFLATED

    # Specify the start time in UTC for the time period you want Tweets from
    start_year, start_month, start_day = list(map(int, args.start.split("-")))
    start_time = datetime.datetime(start_year, start_month, start_day, 0, 0, 0, 0, datetime.timezone.utc)

    # Specify the end time in UTC for the time period you want Tweets from
    end_year, end_month, end_day = list(map(int, args.stop.split("-")))
    end_time = datetime.datetime(end_year, end_month, end_day, 0, 0, 0, 0, datetime.timezone.utc)

    # This is where we specify our query as discussed in module 5
    for emoji in emojis:
        out_dir = os.path.join(args.output, f"{em.demojize(emoji).strip(':')}")
        os.mkdir(out_dir)
        query = f"\{emoji}  lang:en -is:retweet"

        print(f"Current query: {query}")
        try:
            # The search_all method call the full-archive search endpoint to get Tweets based on the query, start and end times
            search_results = client.search_all(query=query, start_time=start_time, end_time=end_time)

            # Twarc returns all Tweets for the criteria set above, so we page through the results
            for page_idx, page in enumerate(search_results):
                print(f"Page: {page_idx}")
                # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
                # so we use expansions.flatten to get all the information in a single JSON
                result = expansions.flatten(page)
                filename = f"{em.demojize(emoji).strip(':')}_start{args.start}_stop{args.stop}_{page_idx}.json"
                with jsonlines.open(filename, mode='w') as writer:
                    for counter, tweet in enumerate(result):
                        tweet["text"] = tweet["text"].encode("utf-16", "surrogatepass").decode("utf-16")
                        writer.write(tweet)
                        if counter == 100:
                            print(f"Page {page_idx} flushed {emoji}")
                            break
                zf = zipfile.ZipFile(f"{os.path.join(out_dir, filename)}.zip", mode="w")
                zf.write(filename, compress_type=compression)
                zf.close()
                os.remove(filename)
                if page_idx == 100:
                    break
        except HTTPError as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Request twitter search API for tweets with emojis "
                    "from context-free dataset starting and ending in a precised timeframe")
    parser.add_argument('--output', default='/scratch/czestoch/twitter-api-emojis',
                        help="Path to the output files directory path")
    parser.add_argument('--start', required=True, help="Start date of tweets in format: Y-m-d eg: 2019-10-01")
    parser.add_argument('--stop', required=True, help="End date of tweets in format: Y-m-d eg: 2019-10-01")
    args = parser.parse_args()

    main(args)
