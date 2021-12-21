import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

project_dir = os.path.join(os.path.dirname(__file__))
EMBEDDINGS_PATH = os.path.join(project_dir, "data", "glove-twitter-200-ambiguity.bin")
EMOJI_CATEGORIZED = os.path.join(project_dir, "data", "emoji_categories.pkl")
EMOJI_IMGS = os.path.join(project_dir, "data", "emoji_imgs")
FINAL_DF = os.path.join(project_dir, "data", "final_dataset.csv.gz")
