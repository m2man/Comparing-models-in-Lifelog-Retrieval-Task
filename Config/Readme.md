# Config Name and its Models
In this project we tried modifying [HADA](https://github.com/m2man/HADA-LAVIS) fusing function and finetuned on Lifelog [LLQA](https://link.springer.com/chapter/10.1007/978-3-030-98358-1_18) dataset. We also compared different retrieval models performance on [LSC'21](http://lifelogsearch.org/lsc/2021/) dataset.

| Config's Name | Models |
| :---: | :---: |
| `Pretrain_Flickr_NoFineTune` | HADA-1 (original version)|
| `Pretrain_Flickr_NoFineTune_Both` | HADA-2 (modified function)|
| `Pretrain_Flickr_5_cont_Both` | HADA-2-FT (finetuned on [LLQA dataset](https://link.springer.com/chapter/10.1007/978-3-030-98358-1_18))|
