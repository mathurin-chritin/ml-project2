# Embeddings generation

The present embeddings have been generated using [GloVe](https://nlp.stanford.edu/projects/glove/).

Follow this procedure to generate them from the original dataset located in `../twitter-dataset.zip` :

1. Download the GloVe processor and copy the required file in the directory :

```bash
git clone https://github.com/stanfordnlp/GloVe glove

export DATASET_PATH=../twitter-datasets  # modify accordingly
cp run_tweets.sh "$DATASET_PATH"/train_neg_full.txt "$DATASET_PATH"/train_pos_full.txt "$DATASET_PATH"/test_data.txt glove/
```

2. Apply stemming and lemmatization on the original dataset :

```bash
# NOTE : you might need to download nltk data with nltk.download('wordnet')
python3 preprocess_stem.py
python3 preprocess_lemmatize.py
```
This should generate two additionnal file for each of the positive and negative tweets, `glove/train_{pos/neg}_full_stemmed.txt` and `glove/train_{pos/neg}_full_lemmatized.txt`.

Then, concatenate each of the set (`.txt`, `_stemmed.txt`, `_lemmatized.txt`) in a single file :


```bash

cat "twitter-datasets/train_neg_full.txt" "twitter-datasets/train_pos_full.txt" > "twitter-datasets/glove/all_tweets_full.txt"
cat twitter-datasets/test_data.txt | cut -d , -f 2 >> "twitter-datasets/glove/all_tweets_full.txt"

cat "twitter-datasets/train_neg_full_stemmed.txt" "twitter-datasets/train_pos_full_stemmed.txt" > "twitter-datasets/glove/all_tweets_full_stemmed.txt"
cat twitter-datasets/test_data_stemmed.txt | cut -d , -f 2 >> "twitter-datasets/glove/all_tweets_full_stemmed.txt"

cat "twitter-datasets/train_neg_full_lemmatized.txt" "twitter-datasets/train_pos_full_lemmatized.txt" > "twitter-datasets/glove/all_tweets_full_lemmatized.txt"
cat twitter-datasets/test_data_lemmatized.txt | cut -d , -f 2 >> "twitter-datasets/glove/all_tweets_full_lemmatized.txt"

```
3. Finally, build the embeddings for each separate set :

```bash
cd glove && chmod +x run_tweets.sh && ./run_tweets.sh
```

You now have built one vocabulary file and one embedding file for each subset ! The files already present on this repository can already be used to run the models.
