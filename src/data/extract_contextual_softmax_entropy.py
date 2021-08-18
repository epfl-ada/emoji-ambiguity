import argparse
import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
from emoji import demojize
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from pandarallel import pandarallel
from scipy.stats import entropy
pandarallel.initialize(nb_workers=8)
tqdm.pandas()

from settings import AMBIGUITY_CLUSTER

from src.data.utils import save_to_csv


def preprocess_tweets(group, replacement='<mask>'):
    emoji = group.emojis.unique()[0]
    if emoji == "*⃣" or emoji == '*️⃣':
        emoji = f"\{emoji}"
    try:
        group.tweet = group.tweet.apply(lambda x: x.replace(emoji, replacement, 1))
        group.tweet = group.tweet.apply(demojize)
    except Exception as e:
        return np.nan
    return group

def get_emoji_softmax_entropy(texts):
    try:
        input_ = tokenizer(texts.tolist(), return_tensors='pt', padding=True, truncation=True)
        mask_index = torch.where(input_["input_ids"] == tokenizer.mask_token_id)[1]
        logits = model(**input_).logits
        softmax = F.softmax(logits, dim=-1)
        masked_word_preds = softmax[torch.arange(softmax.size(0)), mask_index].detach().numpy()
        return np.mean(entropy(masked_word_preds, axis=1))
    except Exception as e:
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate mean entropy of softmax in tweets per emoji')
    parser.add_argument('--input', required=True,
                        help="Path to the cleaned and equalized emoji tweets dataset from kaggle: /scratch/czestoch/cleaned_equalized_tweets.csv.gz")
    parser.add_argument('--output', required=True, help="Path to the output mean entropy of softmax emoji file .csv")
    args = parser.parse_args()

    print("Load data...")
    tweets = pd.read_csv(args.input, header=0, lineterminator='\n', encoding='utf-8')

    print("Preprocess tweets...")
    tweets = tweets.groupby("emojis").parallel_apply(preprocess_tweets)
    tweets = tweets.dropna()

    print("Load model...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")

    print("Extracting embeddings...")
    out = tweets.groupby("emojis").tweet.progress_apply(get_emoji_softmax_entropy)
    del tweets
    out = out.dropna()
    out = out.reset_index()

    print("Saving...")
    save_to_csv(out, args.output)
