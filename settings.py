import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

project_dir = os.path.join(os.path.dirname(__file__))
EMOJIVEC_DATA_PATH = os.path.join(project_dir, "data", "raw", "emoji_dataset_prod.csv")
DEMOGRAPHICS_PATH = os.path.join(project_dir, "data", "raw", "demographic_info.csv")
WUS_LOCAL = os.environ.get("WUS_LOCAL")
WUS_CLUSTER = os.environ.get("WUS_CLUSTER")
AMBIGUITY_PATH = os.path.join(project_dir, "data", "interim", "ambiguity_dataset.csv.gz")
AMBIGUITY_CLUSTER = os.environ.get("AMBIGUITY_CLUSTER")
EMBEDDINGS_PATH = os.path.join(project_dir, "data", "interim", "glove-twitter-200-ambiguity.bin")
CLDR_ANNS_PATH = os.path.join(project_dir, "data", "external", "cldr_39_alpha4_en.xml")
CLDR_DISTANCE = os.path.join(project_dir, "data", "interim", "cldr_distance.csv.gz")
AMBIGUITY_VARIATION = os.path.join(project_dir, "data", "interim", "ambiguity_variation.csv.gz")
EMOJI_IMGS = os.path.join(project_dir, "data", "external", "emoji_imgs")
EMOJI_CATEGORIZED = os.path.join(project_dir, "data", "external", "emoji_categories.pkl")
E2V_EMBEDDINGS = os.path.join(project_dir, "data", "raw", "embeddings", "word2vec", "em_dataset", "emoji2vec.bin")
E2V_MAPPING = os.path.join(project_dir, "data", "raw", "embeddings", "word2vec", "em_dataset", "mapping.pk")

