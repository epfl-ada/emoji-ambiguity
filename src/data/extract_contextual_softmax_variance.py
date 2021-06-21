import argparse
import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
from emoji import demojize
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8)
tqdm.pandas()

from settings import AMBIGUITY_CLUSTER

from src.data.utils import save_to_csv


def preprocess_tweets(group):
    emoji = group.emojis.unique()[0]
    if emoji == "*⃣" or emoji == '*️⃣':
        emoji = f"\{emoji}"
    try:
        group.tweet = group.tweet.apply(lambda x: x.replace(emoji, "<mask>", 1))
        group.tweet = group.tweet.apply(demojize)
    except Exception as e:
        return np.nan
    return group

def get_emoji_softmax_variance(texts):
    try:
        tokenized = [tokenizer.tokenize(text) for text in texts.tolist()]
        tokenized = list(filter(lambda x: "<mask>" in x, tokenized))
        input_ = tokenizer(tokenized, return_tensors='pt', is_split_into_words=True, padding=True, truncation=True)
        mask_index = torch.where(input_["input_ids"] == tokenizer.mask_token_id)[1]
        output = model(**input_)
        logits = output.logits
        softmax = F.softmax(logits, dim=-1)
        over_our_words = softmax[torch.arange(softmax.size(0)), mask_index][:, indices].detach().numpy()
        return np.sum(np.var(over_our_words, 0))
    except Exception as e:
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate variance of softmax over tokens in tweets per emoji')
    parser.add_argument('--input', required=True,
                        help="Path to the downsampled emoji tweets dataset from kaggle: /scratch/czestoch/sampled_tweets_bigger.txt.gz")
    parser.add_argument('--output', required=True, help="Path to the output contextualized emoji variance file .csv")
    args = parser.parse_args()

    print("Load data...")
    # path = "/scratch/czestoch/sampled_tweets_bigger.txt.gz"
    tweets = pd.read_csv(args.input, header=0, lineterminator='\n', encoding='utf-8')

    our_emojis = pd.read_csv(AMBIGUITY_CLUSTER, encoding='utf-8').emoji.unique()
    tweets = tweets[tweets.emojis.isin(our_emojis)]

    df = tweets.groupby("emojis").count()
    numerous_emojis = df[df.tweet >= 100].index.tolist()
    tweets = tweets[tweets.emojis.isin(numerous_emojis)]

    del our_emojis
    del numerous_emojis
    del df

    print("Preprocess tweets...")
    tweets = tweets.groupby("emojis").parallel_apply(preprocess_tweets)
    tweets = tweets.dropna()

    print("Load model...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
    # get indices of emoji words from out data in bert vocabulary
    vocab = list(tokenizer.encoder.keys())
    our_words = set(pd.read_csv(AMBIGUITY_CLUSTER).word.unique())
    indices = []
    for vocab_idx, vocab_word in enumerate(vocab):
        if vocab_word in our_words:
            indices.append(vocab_idx)
    indices = np.array(indices)

    print("Extracting embeddings...")
    out = tweets.groupby("emojis").tweet.progress_apply(get_emoji_softmax_variance)
    del tweets
    out = out.dropna()
    out = out.reset_index()

    print("Saving...")
    # save_to_csv(out, "/scratch/czestoch/softmax_emojis_variances.csv")
    save_to_csv(out, args.output)
