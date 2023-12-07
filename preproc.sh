#!/bin/bash

# python preprocess.py

prefix="_full_stemmed.txt"

cat "twitter-datasets/train_neg$prefix" "twitter-datasets/train_pos$prefix" > "twitter-datasets/glove/all_tweets$prefix"
cat twitter-datasets/test_data_stemmed.txt | cut -d , -f 2 >> "twitter-datasets/glove/all_tweets$prefix"
