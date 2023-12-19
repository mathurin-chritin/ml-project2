# Input folder

This folder holds the preprocessed data for our model to use. A subfolder is there for each of our models, the `run.ipynb` one (CNN) will use the `cnn/` subfolder as input.

## Download preprocessed dataset

To speed up a quick run of `run.ipynb`, download and unzip this file (link[`TODO`]()).

You should have a structure like this :
```bash
input/cnn/
├── X_T_3CH.pt  # this holds the unlabeled data we need to predict
├── X_test_cnn_new_{0-15}_full_3CH.pt
├── X_train_cnn_new_{0-15}_full_3CH.pt
├── y_test_cnn_new_{0-15}_full.pt
└── y_train_cnn_new_{0-15}_full.pt
```

