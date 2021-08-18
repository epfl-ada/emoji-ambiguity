"""
Create lists of filepaths to process tweets
in batches with multiprocessing
"""
import re
import os

regex = re.compile("twitter_stream_2019_09_*")
rootdir = "/dlabdata1/gligoric/spritzer/tweets_pritzer_sample"
september_dirs = list(filter(lambda x: regex.match(x), os.listdir(rootdir)))

batchlist_path = '/home/czestoch/workspace/emoji-ambiguity/data/external/twitter_batch_input_files'
for dir_idx, twitter_dir in enumerate(september_dirs):
    with open(os.path.join(batchlist_path, f"batch_{dir_idx}.txt"), "wt") as file_handle:
        for root, dirs, files in os.walk(os.path.join(rootdir, twitter_dir)):
            for file in files:
                filepath = os.path.join(root, file)
                if ".json.bz2" in filepath:
                    file_handle.write(filepath+"\n")