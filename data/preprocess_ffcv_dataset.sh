#!/bin/bash
# compile train, test and val splits one after the other. 

# train
python3 preprocess_ffcv_datasets_cifar100_tk288_train.py 

# test
python3 preprocess_ffcv_datasets_cifar100_tk288_test.py

# val
python3 preprocess_ffcv_datasets_cifar100_tk288_val.py