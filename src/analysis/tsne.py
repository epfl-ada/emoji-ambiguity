import argparse
import gensim.downloader as api
import spacy
import pandas as pd
from collections import Counter
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle

pd.set_option('mode.chained_assignment', None)

from settings import AMBIGUITY_CLUSTER, AMBIGUITY_PATH
from src.analysis.embedded import find_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run multicore TSNE to get 2d representation of twitter embeddings with emoji vocabularies')
    parser.add_argument('--output', action='store', required=True,
                        help='Location of the output TSNE .pkl')
    parser.add_argument('--cluster', action='store_true',
                        help="With this flag, cluster's datapath will be used with more resources")
    args = parser.parse_args()

    ambiguity_path = AMBIGUITY_PATH
    num_cpus = 4 
    if args.cluster:
        ambiguity_path = AMBIGUITY_CLUSTER
        num_cpus = 8

    print("Loading data...")
    word_embeddings = api.load('glove-twitter-200')
    twitter_space = {"word": [], "embedding": [], "space": "twitter"}
    for word in word_embeddings.vocab.keys():
        twitter_space["word"].append(word)
        twitter_space["embedding"].append(word_embeddings.get_vector(word))
    twitter_space = pd.DataFrame(twitter_space)
    emojis = pd.read_csv(ambiguity_path, encoding='utf-8')

    print("Choosing subset of words...")
    vocabularies = emojis[["emoji", "word"]].groupby("emoji").word.apply(list).apply(Counter)\
                                            .reset_index().rename({"word": "vocabulary"}, axis=1)\
                                            .set_index("emoji")
    ambiguity_vocab = {st for row in vocabularies.vocabulary for st in row}
    print("Computing embeddings...")
    tokenizer = spacy.load("en_core_web_sm")
    emoji_subspace = {"word": [], "embedding": [], "space": "emoji"}
    for emoji_description in ambiguity_vocab:
        tokens = {token.text for token in tokenizer(emoji_description)}
        vec = find_embedding(tokens, word_embeddings)
        if vec is not None:
            emoji_subspace["word"].append(emoji_description)
            emoji_subspace["embedding"].append(vec)
    emoji_subspace = pd.DataFrame(emoji_subspace)

    whole_space = pd.concat((twitter_space, emoji_subspace))
    del word_embeddings

    print("Preparing numpy representation...")
    X_ls = whole_space.embedding.tolist()
    X_np = []
    for embedding in tqdm(X_ls):
        X_np.append(list(embedding))
    del X_ls
    X_np = np.array(X_np)

    print("Running preliminary PCA...")
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(X_np)
    print("TSNE in progress...")
    whole_space[['tsne_x', 'tsne_y']] = TSNE(n_components=2, n_jobs=num_cpus).fit_transform(reduced)
    del reduced
    print("Saving output...")
    with open(args.output, "wb") as f:
        pickle.dump(whole_space, f)




