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
from src.data.extract_contextual_softmax_entropy import preprocess_tweets


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
                        help="Path to the cleaned and equalized emoji tweets dataset from kaggle: /scratch/czestoch/cleaned_equalized_tweets.csv.gz")
    parser.add_argument('--output', required=True, help="Path to the output contextualized emoji variance file .csv")
    args = parser.parse_args()

    print("Load data...")
    tweets = pd.read_csv(args.input, header=0, lineterminator='\n', encoding='utf-8')

    print("Preprocess tweets...")
    # replace all emojis with a special token to mitigate the influence
    # of wordpiece model on OOV emojis
    emoji_token = "[EMOJI]"
    tweets = tweets.groupby("emojis").parallel_apply(lambda x: preprocess_tweets(x, replacement=emoji_token))
    tweets = tweets.dropna()

    print("Load model...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",\
                                            additional_special_tokens=[emoji_token])
    model = AutoModel.from_pretrained("vinai/bertweet-base")
    model.resize_token_embeddings(len(tokenizer))

    print("Extracting embeddings...")
    variances = tweets.groupby("emojis").tweet.progress_apply(get_embeddings_variance)
    variances = variances.dropna()
    variances = variances.reset_index().rename({0: "variance"}, axis=1)
    
    print("Saving...")
    save_to_csv(variances, args.output)
