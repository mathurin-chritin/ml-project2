# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

EMBEDDINGS = "twitter-datasets/glove/embeddings.txt"
VOCAB = "twitter-datasets/glove/vocab.txt"
# embeddings = np.load("embeddings.npy")
# with open("vocab_cut.txt", "r") as f:
#     vocab = {line.rstrip(): idx for idx, line in enumerate(f)}


vecs = {}
with open(EMBEDDINGS, "r") as f:
    for line in f:
        pline = line.rstrip().split(' ')
        word = pline[0]
        vecs[word] = np.array([float(x) for x in pline[1:]])

with open(VOCAB, "r") as f:
    vocab = {x.rstrip().split(' ')[0]: i for i,x in enumerate(f)}

embeddings = np.zeros((len(vocab), len(vecs[list(vecs.keys())[0]])))

for w, v in vecs.items():
    if w == "<unk>":
        continue
    embeddings[vocab[w], :] = v

# %%
with open("twitter-datasets/train_neg.txt", "r") as f:
    n_tweets = [line.rstrip().split() for line in f]

with open("twitter-datasets/train_pos.txt", "r") as f:
    p_tweets = [line.rstrip().split() for line in f]

# %%
df = pd.DataFrame(columns=[f'f{x+1}' for x in range(embeddings.shape[1])]+['label'])
series = [df]

def load_tweets(tweets_file, series, label):
    i = 0
    tot = len(tweets_file)
    for tweet in tweets_file:
        if i%500 == 0:
            print(f"{i}/{tot} ({int(i/tot*100)} %)")
        indices = [vocab.get(word, -1) for word in tweet if word in vocab.keys()]
        if len(indices) == 0:
            i+=1
            continue
        tweet_embedding = np.mean(np.array([embeddings[indice] for indice in indices]), axis=0)
        serie_dict = {f'f{x+1}': data for x, data in enumerate(tweet_embedding)}
        serie_dict['label'] = label
        series.append(pd.Series(serie_dict).to_frame().T)
        i+=1
    return series

series = load_tweets(p_tweets, series, 1)
series = load_tweets(n_tweets, series, -1)

#%%
print("concatenating...")
df = pd.concat(series, ignore_index=True)
print(df.head())

# %%
RANDOM_SEED = 1234
# shuffle the dataframe
df = df.sample(n=df.shape[0], random_state=RANDOM_SEED)
X = df[df.columns[:-1]]
y = df[df.columns[-1]]
# use the function from sklearn to split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

# %%
