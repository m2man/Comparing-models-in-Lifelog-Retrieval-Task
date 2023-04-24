#!/bin/bash

# python main.py -cp Config/Pretrain_Flickr_7.yml -rm train
# python main.py -cp Config/Pretrain_Flickr_7_cont_Both.yml -rm train
# python run_encode_lsc.py -cp Config/Pretrain_Flickr_2_new_loss_cont_Both_smallDO.yml -rm both
# python run_encode_lsc.py -cp Config/Pretrain_Flickr_2_new_loss_cont_Both_smallLR.yml -rm both
python run_encode_lsc.py -cp Config/Pretrain_Flickr_NoFinetune_Both.yml -rm both
# python run_encode_lsc.py -cp Config/Pretrain_Flickr_NoFinetune.yml -rm both