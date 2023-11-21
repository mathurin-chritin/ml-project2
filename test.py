# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeClassifier, SGDClassifier, LinearRegression, LogisticRegression
from sklearn.metrics import f1_score


EMBEDDINGS = "embeddings.txt"
VOCAB = "vocab.txt"

POS_TWEETS = "twitter-datasets/train_pos.txt"
NEG_TWEETS = "twitter-datasets/train_neg.txt"
TEST_DATA = "twitter-datasets/test_data.txt"

# parse embeddings
vecs = {}
with open(EMBEDDINGS, "r") as f:
    for line in f:
        pline = line.rstrip().split(' ')
        word = pline[0]
        vecs[word] = np.array([float(x) for x in pline[1:]])

# parse vocabulary and build an index
with open(VOCAB, "r") as f:
    vocab = {x.rstrip().split(' ')[0]: i for i,x in enumerate(f)}

embeddings = np.zeros((len(vocab), len(vecs[list(vecs.keys())[0]])))
for w, v in vecs.items():
    if w == "<unk>":
        continue
    embeddings[vocab[w], :] = v

# %%
with open(NEG_TWEETS, "r") as f:
    n_tweets = [line.rstrip().split() for line in f]
with open(POS_TWEETS, "r") as f:
    p_tweets = [line.rstrip().split() for line in f]

#%%
testing_tweets = []
testing_tweets_ids = []
with open(TEST_DATA, "r") as f:
    for line in f:
        parsed_line = line.rstrip().split(',')
        testing_tweets.append(','.join(parsed_line[1:]).split())
        testing_tweets_ids.append(int(parsed_line[0]))

# %%
# convert a tweet to an embedding of shape (20,) which is the mean of each embedding of each word.
series_train = []
series_test = []

def load_tweets(tweets_list, series, label=None):
    print("Loading tweets...")
    i = 0
    tot = len(tweets_list)
    for tweet in tweets_list:
        if i%1000 == 0:
            print(f"{i}/{tot} ({int(i/tot*100)} %)")
        indices = [vocab.get(word, -1) for word in tweet if word in vocab.keys()]
        if len(indices) == 0:
            tweet_embedding = np.zeros((20,))
        else:
            tweet_embedding = np.mean(embeddings[indices], axis=0)
        serie_dict = {f'f{x+1}': data for x, data in enumerate(tweet_embedding)}
        if label is not None:
            serie_dict['label'] = label
        series.append(pd.Series(serie_dict))
        i+=1
    return series

# add both negative and positive tweets, will be shuffled later
series_train = load_tweets(p_tweets, series_train, 1)
series_train = load_tweets(n_tweets, series_train, -1)

# no label since this is the prediction set
series_test = load_tweets(testing_tweets, series_test)

#%%
# use DataFrames to represent data
print("Creating DataFrame...")
df_train = pd.DataFrame(series_train)

df_test = pd.DataFrame(series_test)
df_test["index"] = testing_tweets_ids
df_test.set_index(['index'], inplace=True) # keep indexes as in the input file


# print last 5 columns of the DataFrames (df_test has no "label" column)
print("Training DataFrame sample")
print(df_train[df_train.columns[-5:]].head(5))
print("Testing DataFrame sample")
print(df_test[df_test.columns[-5:]].head(5))

# %%
RANDOM_SEED = 1234
# shuffle the dataframe
df_train = df_train.sample(n=df_train.shape[0], random_state=RANDOM_SEED)
X = df_train[df_train.columns[:-1]]
y = df_train[df_train.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)


# %%
linear = LinearRegression(copy_X=True)
linear.fit(X_train, y_train)
weights, intercept = linear.coef_, linear.intercept_

# %%
ridge = RidgeClassifier(alpha=0.001, copy_X=True, max_iter=5000)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print(f1_score(y_test, y_pred))

# %%
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, max_iter=1000, random_state=RANDOM_SEED+1)
sgd.fit(X_train, y_train, coef_init=weights, intercept_init=intercept)
y_pred = sgd.predict(X_test)
print(f1_score(y_test, y_pred))

# %%
logistic = LogisticRegression(penalty='l2', random_state=RANDOM_SEED+4, max_iter=100)
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print(f1_score(y_test, y_pred))

# %%
# make the predictions on the testing set
predictions = logistic.predict(df_test)
df_predictions = pd.DataFrame({"Id": df_test.index,
                               "Prediction": predictions},
                               dtype=int)

# %%
prediction_file = "my_predictions.csv"
print(f"Saving to {prediction_file}")
df_predictions.to_csv(prediction_file, index=False)

# %%
