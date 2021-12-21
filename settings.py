import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

project_dir = os.path.join(os.path.dirname(__file__))
EM_DATASET = os.path.join(project_dir, "data", "raw", "emoji_dataset_prod.csv")
DEMOGRAPHICS_PATH = os.path.join(project_dir, "data", "raw", "demographic_info.csv")
AMBIGUITY_PATH = os.path.join(project_dir, "data", "processed", "ambiguity_dataset.csv.gz")
AMBIGUITY_CLUSTER = os.environ.get("AMBIGUITY_CLUSTER")
EMBEDDINGS_PATH = os.path.join(project_dir, "data", "external", "glove-twitter-200-ambiguity.bin")
EMBEDDINGS_CLUSTER = os.environ.get("EMBEDDINGS_CLUSTER")
AMBIGUITY_VARIATION = os.path.join(project_dir, "data", "processed", "ambiguity_variation.csv.gz")
EMOJI_IMGS = os.path.join(project_dir, "data", "external", "emoji_imgs")
EMOJI_IMGS_CLUSTER = os.environ.get("EMOJI_IMGS_CLUSTER")
EMOJI_CATEGORIZED = os.path.join(project_dir, "data", "external", "emoji_categories.pkl")
TWITTER_COUNTS = os.path.join(project_dir, "data", "external", "twitter-api-context-free-emoji-counts.csv.gz")
TWITTER_TOKEN = os.environ.get("TWITTER_TOKEN")
FINAL_DF = os.path.join(project_dir, "data", "processed", "final_dataset.csv.gz")
BASELINE_VARIATION = 0.6860900973280272
BASELINE_VARIATION_CIs = [0.5707602351453775, 0.8834659105717826]
BASELINE_VOCAB_SIZE = 30
BASELINE_VOCAB_SIZE_CIs = [27.0, 30.0]
