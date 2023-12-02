#%%
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import math

#%%
# lire données 

EMBEDDINGS = "embeddings_stemmed.txt"
VOCAB = "vocab_stemmed.txt"

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

with open(NEG_TWEETS, "r") as f:
    n_tweets = [line.rstrip().split() for line in f][:100]
with open(POS_TWEETS, "r") as f:
    p_tweets = [line.rstrip().split() for line in f][:100]

# Stack the two lists together (will be used to see max_length of tweet)
combined_tweets = n_tweets + p_tweets

#%%


testing_tweets = []
testing_tweets_ids = []
with open(TEST_DATA, "r") as f:
    for line in f:
        parsed_line = line.rstrip().split(',')
        testing_tweets.append(','.join(parsed_line[1:]).split())
        testing_tweets_ids.append(int(parsed_line[0]))


# convert a tweet to an embedding of shape (max_length,) which is the length of maximal tweet, so added padding
series_train=[]
series_test=[]
length_list=[]
def modified_load_tweets(tweets_list, series, max_tweet_length, label=None ):
    print("Loading tweets...")
    i = 0
    tot = len(tweets_list)
    
    for tweet in tweets_list:
        if i % 1000 == 0:
            print(f"{i}/{tot} ({int(i/tot*100)} %)")

        
       
        embeddings_list = [embeddings[vocab.get(word)] for word in tweet if word in vocab.keys()] # have embedding of each word of tweet
        embeddings_list_torch = torch.FloatTensor(embeddings_list)
        length_tweet=len(embeddings_list)
        length_list.append(length_tweet)
        
        
        diff_length=abs(length_tweet-max_tweet_length)
        
        if len(embeddings_list) == 0:
            torch.zeros((max_tweet_length, len(vecs[list(vecs.keys())[0]])))
        else:
           tweet_mean = torch.mean(embeddings_list_torch, axis=0)
           tweet_embeddings = torch.ones((max_tweet_length, len(vecs[list(vecs.keys())[0]])))*tweet_mean # to have all tweets of shape (#tweets, max_tweet_len, 20)

           if (diff_length%2==0):
               tweet_embeddings[int(diff_length/2):int(max_tweet_length-diff_length/2),:] = embeddings_list_torch #putting them in the middle to do some kind of padding
           else:
               tweet_embeddings[math.floor(diff_length/2):(max_tweet_length-math.ceil(diff_length/2)),:] = embeddings_list_torch
        if label is not None:
            tweet_embeddings = torch.vstack((tweet_embeddings,label*torch.ones(20,)))
        series.append(tweet_embeddings)
        i += 1
    
    return series
#%%
max_length_train = max(len(tweet) for tweet in combined_tweets)
max_length_test = max(len(tweet) for tweet in testing_tweets)
max_tot=max(max_length_train,max_length_test)



# add both negative and positive tweets, will be shuffled later
series_train = modified_load_tweets(p_tweets, series_train, max_tot, 1)
series_train = modified_load_tweets(n_tweets, series_train, max_tot, -1)

# no label since this is the prediction set
series_test = modified_load_tweets(testing_tweets, series_test, max_tot)
#%%


#use torch to represent data
torch_series_train = torch.stack(series_train)
torch_series_test = torch.stack(series_test)
X = torch_series_train[:,:-1]
y = torch_series_train[:,-1]
RANDOM_SEED = 1234

X_flattened = X.view(X.size(0), -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
X_train = X_train.view(X_train.size(0), max_tot, 20)
X_test = X_test.view(X_test.size(0), max_tot, 20)
torch.save(X_train, 'X_train_cnn_new.pt')
torch.save(X_test, 'X_test_cnn_new.pt')
torch.save(y_train, 'y_train_cnn_new.pt')
torch.save(y_test, 'y_test_cnn_new.pt')


# %%
