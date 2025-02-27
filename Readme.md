# Comparison between Concept-based and Embedding-based models in Lifelog Retrieval Task

## Introduction
There have been many search engines used in Lifelog Retrieval Task ranging from traditional concept-based model to multimodal embedding models (BLIP, CLIP, or HADA). In this repo, we did an comparison in the automatic manner between concept-based and many SOTA multimodal embedding models such as BLIP, CLIP, and HADA.

The data we used in the experiment was [LSC'21 data](http://lifelogsearch.org/lsc/2021/) with [23 official topics in LSC'21](http://lifelogsearch.org/lsc/resources/lsc21-topics-qrels-shared.txt).

## Installation
Please follow [HADA repo](https://github.com/m2man/HADA-LAVIS) for the installation.

We used **mlflow-ui** to keep track the performance between configurations. Please modify or remove this related-part if you do not want to use.

## Pretrained Models and Encoded Features
We uploaded the pretrained models and encoded features [here](https://drive.google.com/drive/folders/17tajfdA0TofKL0ohV7qZnu3JN4Tt0f4B?usp=share_link). Please download and put the pretrain models in **Output/llqa** folder, encoded features in **EncodedLifelog** folder.

### List of Models
| Config's Name | Models |
| :---: | :---: |
| `Pretrain_Flickr_NoFineTune` | HADA-1 (original version)|
| `Pretrain_Flickr_NoFineTune_Both` | HADA-2 (modified function)|
| `Pretrain_Flickr_5_cont_Both` | HADA-2-FT (finetuned on [LLQA dataset](https://link.springer.com/chapter/10.1007/978-3-030-98358-1_18))|

## Train and Encode Feature
If you want to train HADA models from scratch, remember to update the path in the config files in **Config** folders. Then you can train by the file `run_bash.sh`

You can run `run_encode_lsc.py` to encode images and queries after training models (included in `run_bash.sh`).
```python
# HADA-1 
python run_encode_lsc.py -cp Config/Pretrain_Flickr_NoFinetune.yml -rm both

# HADA-2 (this will include BLIP and CLIP in the output)
python run_encode_lsc.py -cp Config/Pretrain_Flickr_NoFinetune_Both.yml -rm both

# HADA-2-FT
python run_encode_lsc.py -cp Config/Pretrain_Flickr_5_cont_Both.yml -rm both
```

## Evaluate
You can run jupyter files: `Evaluate_BLIP_CLIP_Lifelog.ipynb` and `Evaluate_HADA_Lifelog.ipynb`

## Query Batches in Interactive Manner
We also conducted an interactive experiment where we asked 8 volunteers to used an interactive lifelog retrieval system (with 2 different backend search engines: concept-based and embedding-based) to solve LSC'21 queries. We divided queries into 4 smaller query batches (6 queries / batch), each volunteer used system with each search engine to solve 1 batch (2 batch / user in total).

The interactive system we used was [E-MyScéal](https://dl.acm.org/doi/10.1145/3512729.3533012). The backend were concept-based [MyScéal 2.0](https://dl.acm.org/doi/abs/10.1145/3463948.3469064) and embedding-based HADA. The result and the batches are in **interactive_data** folder.

## Contact
For any issue or comment, you can directly email me at manh.nguyen5@mail.dcu.ie