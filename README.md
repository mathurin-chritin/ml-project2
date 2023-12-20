# Machine Learning - Tweet Classification

    Pedro Pino 310030
    Mia Zosso 300880
    Mathurin Chritin 288065

---

This repository holds the code for model on the Tweet Classification problem.

### Task Introduction

The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

*Working with Twitter data:* We provide a large set of training tweets, one tweet per line. All tweets in the file train pos.txt (and the train pos full.txt counterpart) used to have positive smileys, those of train neg.txt used to have a negative smiley. Additionally, the file test data.txt contains 10’000 tweets without any labels, each line numbered by the tweet-id.

Note that all tweets have already been pre-processed so that all words (tokens) are separated by a single whitespace. Also, the smileys (labels) have been removed.

### Run the model

Our model takes **preprocessed** data located under `input/cnn/` :

```bash
input/cnn/
├── X_T_3CH.pt  # this holds the unlabeled data we need to predict
├── X_test_cnn_new_{0-15}_full_3CH.pt
├── X_train_cnn_new_{0-15}_full_3CH.pt
├── y_test_cnn_new_{0-15}_full.pt
└── y_train_cnn_new_{0-15}_full.pt
```

You can refer to the README inside `input/`, but the preprocessed data we generated for this project can be directly downloaded at this [link](TODO). This avoids to run the quite expensive preprocessing pipeline we designed for this project.


Once you downloaded and unzipped the preprocessed data as in the above structure, refer to `run.ipynb` to train our best model and generate a prediction file.

