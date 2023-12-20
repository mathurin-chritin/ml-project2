# Input folder

This folder holds the preprocessed data for our model to use. A subfolder is there for each of our models, the `run.ipynb` one (CNN) will use the `cnn/` subfolder as input.

## Download preprocessed dataset

### Notebook: `run.ipynb`

To speed up a quick run of `run.ipynb`, download and unzip this file (link[`https://drive.google.com/file/d/1vMLgzSZ1Wxpj9YNIGNamC2yRysJ41awG/view?usp=drive_link`]()).

You should have a structure like this :
```bash
input/cnn/
├── X_T_3CH.pt                          # unlabeled data on which we will make predictions
├── X_test_cnn_new_{0-15}_full_3CH.pt   # X matrice for validation set
├── X_train_cnn_new_{0-15}_full_3CH.pt  # X matrice for training set
├── y_test_cnn_new_{0-15}_full.pt       # y matrice for validation set (labels)
└── y_train_cnn_new_{0-15}_full.pt      # y matrice for training set (labels)
```

### (optional) Notebook: `fully_connected.ipynb`

For the fully connected the subfolder holding the preprocessed data is available in this link[https://drive.google.com/file/d/1gO1ez3z78aEwszjmFUeyGa7y6nKScoxc/view?usp=sharing], running the notebook fully_connected.ipynb will also create the .csv necessary files but this would take long time. The cell where this files are created is clearly stated.
```bash
input/fc/
├── xtestpred.csv   
├── X_test.csb
├── X_train.csv
├── y_test.csv
└── y_train.csv
```
