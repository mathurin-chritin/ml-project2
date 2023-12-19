# %%
from nltk.stem.snowball import EnglishStemmer


POS_TWEETS = "glove/train_pos_full.txt"
NEG_TWEETS = "glove/train_neg_full.txt"
TEST_DATA = "glove/test_data.txt"

#%%
with open(NEG_TWEETS, "r") as f:
    n_tweets = [line.rstrip().split() for line in f]
with open(POS_TWEETS, "r") as f:
    p_tweets = [line.rstrip().split() for line in f]

with open(TEST_DATA, "r") as f:
    testing_tweets = [line.rstrip().split() for line in f]


# %%
stemmer = EnglishStemmer()
def stem(tweets):
    return [[stemmer.stem(w) for w in tweet] for tweet in tweets]
    
#%%
print("Stemming")
p_tweets = stem(p_tweets)
print("Stemming")
n_tweets = stem(n_tweets)
print("Stemming")
testing_tweets = stem(testing_tweets)


# %%
with open(POS_TWEETS.replace(".txt", "_stemmed.txt"), "w+") as f:
    f.write('\n'.join(([' '.join(tweet) for tweet in p_tweets])))
with open(NEG_TWEETS.replace(".txt", "_stemmed.txt"), "w+") as f:
    f.write('\n'.join(([' '.join(tweet) for tweet in n_tweets])))
with open(TEST_DATA.replace(".txt", "_stemmed.txt"), "w+") as f:
    f.write('\n'.join(([' '.join(tweet) for tweet in testing_tweets])))


