import argparse
import pandas as pd

from settings import EM_DATASET, AMBIGUITY_PATH
from src.data.utils import is_valid_file, save_to_csv

# example usage
# python3 em2ambiguity.py
# --em-dataset data/raw/emoji_dataset_prod.csv
# --output data/interim/ambiguity_dataset.csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fix of mistakes spotted during analysis of em dataset')
    parser.add_argument('--em-dataset', default=EM_DATASET,
                        type=lambda x: is_valid_file(parser, x),
                        help='Location of the emoji em dataset')
    parser.add_argument('--output', default=AMBIGUITY_PATH,
                        help='Location of the output ambiguity csv')
    args = parser.parse_args()

    print("Reading data...")
    emojis = pd.read_csv(args.em_dataset, encoding='utf-8')

    vocabularies = emojis[["emoji", "word"]] \
        .groupby("emoji") \
        .word.apply(list) \
        .reset_index() \
        .rename({"word": "vocabulary"}, axis=1)

    print("Mistakes mapping...")
    # Manually spotted mistakes
    # Other cases that didn't pass dictionary check -> notebook 01_ambiguity_exploration
    # Fix of artifacts from emojivec project postprocessing
    # Spotted during embedding analysis -> 02_variation
    correction_mapping = {"oh h": "ohh", "om g": "omg",
                          "id k": "idk", "hmm m": "hmmm",
                          "ship a": "ship", "jappanesepostoffice": "japanese post office",
                          "gaypriide": "gay pride", "redbeanonigiri": "red bean onigiri",
                          "bent o": "bento", "manor a": "manora", "ta j": "taj",
                          "disappointed d": "disappointed", "fastfoward": "fastforward",
                          "tolietpaper": "toilet paper", "travelling": "traveling", "ha ult": "hault",
                          "fleurdelis": "fleur de lis", "hearingaid": "hearing aid", "nowater": "no water",
                          "nonsmoking": "non smoking", "fireextinguisher": "fire extinguisher"}

    emojis.word = emojis.word.replace(correction_mapping)

    print("Saving to csv...")
    save_to_csv(emojis, args.output)
