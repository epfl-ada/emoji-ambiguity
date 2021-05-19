import argparse
import sys

import pandas as pd
from emoji import get_emoji_regexp
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from transformers import BertTokenizer, BertModel

from settings import AMBIGUITY_PATH


def curry_find_emojis(all_emojis):
    return udf(lambda text: find_emojis(text, all_emojis), ArrayType(StringType()))


def find_emojis(text, all_emojis):
    return list(filter(lambda x: x in all_emojis, get_emoji_regexp().findall(text)))


def curry_get_embedding(model, tokenizer, original_tokenizer_size):
    return udf(lambda text: get_embedding(text, model, tokenizer, original_tokenizer_size),
               ArrayType(ArrayType(FloatType())))


def get_embedding(text, model, tokenizer, original_tokenizer_size):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    # Get only emoji tokens embeddings
    return output[0][:, (encoded_input['input_ids'] >= original_tokenizer_size) \
                            .nonzero(as_tuple=True)[1], :].detach().numpy().squeeze(0).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--input', required=True,
                        help="Path to the input emoji tweet dataset from kaggle: emojitweets-01-04-2018.txt.gz")
    parser.add_argument('--output', required=True, help="Path to the output parquet directory")
    parser.add_argument('--num-cpus', required=True, help="Number of cores to use by spark session")
    args = parser.parse_args(sys.argv[1:])

    # create the session
    spark = SparkSession \
        .builder \
        .appName("Extract BERT embeddings for emojis in tweets") \
        .config('spark.master', f'local[{args.num_cpus}]')\
        .getOrCreate()

    tweets = spark.read.load(args.input, format="csv", sep=",", inferSchema="true", header="true").limit(1000)

    # Initialize BERT model and tokenizer, expand tokenizer with emojis
    all_emojis = set(pd.read_csv(AMBIGUITY_PATH, encoding='utf-8').emoji.unique())
    tweets = tweets.withColumn("emojis", curry_find_emojis(all_emojis)(tweets["tweet"]))
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    original_tokenizer_size = len(tokenizer)
    tokenizer.add_tokens(list(all_emojis))
    model.resize_token_embeddings(len(tokenizer))

    # Get embedding for each tweet
    tweets = tweets.withColumn("embedding",
                               curry_get_embedding(model, tokenizer, original_tokenizer_size)(col("tweet")))
    # Combine emoji and embedding column for each tweet to be able to explode
    combine = udf(lambda x, y: list(zip(x, y)),
                  ArrayType(StructType([StructField("emojis", StringType()),
                                        StructField("embedding", ArrayType(FloatType()))])))
    # Explode each row -> tweet can have multiple emojis
    df = tweets.withColumn("new", combine("emojis", "embedding")).withColumn("new", explode("new")) \
        .select("tweet", col("new.emojis").alias("emoji"), col("new.embedding").alias("embedding"))
    # Change datatype to spark vector, we will use it to calculate covariance matrix later
    func = udf(lambda vs: Vectors.dense(vs), VectorUDT())
    df = df.withColumn("embedding", func("embedding"))

    df.select("emoji", "embedding").write.partitionBy("emoji").format("parquet").save(args.output)
