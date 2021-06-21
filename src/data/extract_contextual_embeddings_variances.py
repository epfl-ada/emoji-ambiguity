import argparse
import pandas as pd
import numpy as np
from emoji import demojize
from transformers import AutoTokenizer, AutoModel
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
        group.tweet = group.tweet.apply(lambda x: x.replace(emoji, "[EMOJI]", 1))
        group.tweet = group.tweet.apply(demojize)
    except Exception:
        return np.nan
    return group

def get_embeddings_variance(group):
    try:
        encoded_input = tokenizer(group.tolist(), return_tensors='pt', padding=True, truncation=True)
        embeddings = model(**encoded_input)[1].detach().numpy()
        return np.sum(embeddings.var(0))
    except Exception:
        return np.nan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate variance of tweets' embeddings per emoji")
    parser.add_argument('--input', required=True,
                        help="Path to the downsampled emoji tweets dataset from kaggle: /scratch/czestoch/sampled_tweets_bigger.txt.gz")
    parser.add_argument('--output', required=True, help="Path to the output contextualized emoji variance file .csv")
    args = parser.parse_args()

    print("Load data...")
    # path = "/scratch/czestoch/sampled_tweets_bigger.txt.gz"
    tweets = pd.read_csv(args.input, header=0, lineterminator='\n', encoding='utf-8')

    print("Preprocess tweets...")
    tweets = tweets.groupby("emojis").parallel_apply(preprocess_tweets)
    tweets = tweets.dropna()

    print("Load model...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",\
                                            additional_special_tokens=["[EMOJI]"])
    model = AutoModel.from_pretrained("vinai/bertweet-base")
    model.resize_token_embeddings(len(tokenizer))

    print("Extracting embeddings...")
    variances = tweets.groupby("emojis").tweet.progress_apply(get_embeddings_variance)
    variances = variances.dropna()
    variances = variances.reset_index().rename({0: "variance"}, axis=1)

    # save_to_csv(variances, "/scratch/czestoch/bert_masked_emojis_embeddings_variances.csv")
    save_to_csv(variances, args.output)

