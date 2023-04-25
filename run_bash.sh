#!/bin/bash

# HADA-1 (BLIP and CLIP included)
python main.py -cp Config/Pretrain_Flickr_NoFinetune.yml -rm train

# HADA-2
python main.py -cp Config/Pretrain_Flickr_NoFinetune_Both.yml -rm train

# HADA-2-FT 
python main.py -cp Config/Pretrain_Flickr_5.yml -rm train
python main.py -cp Config/Pretrain_Flickr_5_cont_Both.yml -rm train

# Encode lifelog images and lifelog queries
python run_encode_lsc.py -cp Config/Pretrain_Flickr_NoFinetune.yml -rm both
python run_encode_lsc.py -cp Config/Pretrain_Flickr_NoFinetune_Both.yml -rm both
python run_encode_lsc.py -cp Config/Pretrain_Flickr_5_cont_Both.yml -rm both
