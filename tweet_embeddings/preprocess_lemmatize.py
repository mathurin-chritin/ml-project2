# %%
from nltk.stem import WordNetLemmatizer


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
lemmatizer = WordNetLemmatizer()
def lemmatize(tweets):
    return [[lemmatizer.lemmatize(w) for w in tweet] for tweet in tweets]
    
#%%
print("Stemming")
p_tweets = lemmatize(p_tweets)
print("Stemming")
n_tweets = lemmatize(n_tweets)
print("Stemming")
testing_tweets = lemmatize(testing_tweets)

# %%
with open(POS_TWEETS.replace(".txt", "_lemmatized.txt"), "w+") as f:
    f.write('\n'.join(([' '.join(tweet) for tweet in p_tweets])))
with open(NEG_TWEETS.replace(".txt", "_lemmatized.txt"), "w+") as f:
    f.write('\n'.join(([' '.join(tweet) for tweet in n_tweets])))
with open(TEST_DATA.replace(".txt", "_lemmatized.txt"), "w+") as f:
    f.write('\n'.join(([' '.join(tweet) for tweet in testing_tweets])))


