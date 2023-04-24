import json
import joblib
import argparse
import numpy as np
from Retrieval_Utils import i2t, t2i, evaluate_recall
from Utils import write_to_file
import torch
import mlflow
import xml.etree.ElementTree as ET
import Utils as ut
from Controller import Controller as Ctr
from Controller_Both import Controller as Ctr_Both
from lavis.models import load_model_and_preprocess, load_model, load_preprocess
from lavis.datasets.builders import load_dataset

IMG_DIR = '/mnt/data/mount_4TBSSD/nmduy/webp_images'

mlflow.set_tracking_uri('http://localhost:1409')

tree = ET.parse('lsc2019-topics.xml')
relevance_images = []
root = tree.getroot()
for child in root:
    for j in child:
        for k in range(len(j[3])):
            relevance_images.append(j[3][k].text)
for query in json.load(open("lsc21_targets.json")):
    relevance_images.extend([image.split('/')[-1] for image in query])
print("Relevance images:", relevance_images[-3:])

def get_index_group(list_images, list_texts):
    li = list_images.copy()
    lt = list_texts.copy()
    list_set_lt = list(set(lt))
    list_set_li = list(set(li))
    index_lt = np.array([list_set_lt.index(x) for x in lt])
    index_li = np.array([list_set_li.index(x) for x in li])
    count = 0
    index_both = np.array([-1] * len(li))
    for idx in range(len(li)):
        if index_both[idx] > -1:
            continue
        idx_i = index_li[idx]
        idx_t = index_lt[idx]
        same_idx_i = np.where(index_li == idx_i)[0]
        same_idx_t = np.where(index_lt == idx_t)[0]
        index_both[same_idx_i] = count
        index_both[same_idx_t] = count
        count = count + 1
    return index_both.tolist()

def get_match_dict(list_images, list_texts, groups=None):
    li = list_images.copy()
    lt = list_texts.copy()
    list_set_li = list(set(li))
    list_set_lt = list(set(lt))
    if groups is None:
        groups = get_index_group(list_images=li, list_texts=lt)
    image_match_dict = {}
    index_image_match_dict = {}
    for idxi, iid in enumerate(list_set_li):
        idx_iid = li.index(iid)
        g = groups[idx_iid]
        match_text = [(ix, x) for ix, x in enumerate(list_set_lt) if groups[lt.index(x)] == g]
        image_match_dict[iid] = [x[1] for x in match_text]
        index_image_match_dict[idxi] = [x[0] for x in match_text]
    text_match_dict = {}
    index_text_match_dict = {}
    for idxt, tid in enumerate(list_set_lt):
        idx_tid = lt.index(tid)
        g = groups[idx_tid]
        match_image = [(ix, x) for ix, x in enumerate(list_set_li) if groups[li.index(x)] == g]
        text_match_dict[tid] = [x[1] for x in match_image]
        index_text_match_dict[idxt] = [x[0] for x in match_image]
    return image_match_dict, text_match_dict, index_image_match_dict, index_text_match_dict

skipped = 0
def read_files(files):
    global skipped
    images = []
    captions = []
    for filename in files:
        with open(f"data/{filename}") as f:
            for line in f.readlines():
                if '\t' not in line.strip():
                    print(line)
                    continue
                image, caption = line.strip().split('\t')
                if image.split('/')[-1] in relevance_images:
                    skipped +=1
                    continue
                images.append(f"{IMG_DIR}/{image.split('.')[0]}.webp")
                captions.append(caption)
    return images, captions

LL_TRAIN_IMAGES, LL_TRAIN_TEXTS = read_files(["clip_short.data", "clip.data"])
LL_VAL_IMAGES, LL_VAL_TEXTS = read_files(["clip_short.data_test"])
LL_TRAIN_GROUP = get_index_group(list_images=LL_TRAIN_IMAGES, list_texts=LL_TRAIN_TEXTS)
LL_VAL_GROUP = get_index_group(list_images=LL_VAL_IMAGES, list_texts=LL_VAL_TEXTS)
VAL_I2T_DICT, VAL_T2I_DICT, VAL_IDX_I2T_DICT, VAL_IDX_T2I_DICT = get_match_dict(list_images=LL_VAL_IMAGES, 
                                                                                list_texts=LL_VAL_TEXTS, 
                                                                                groups=LL_VAL_GROUP)

def get_model_dict(config):
    print(f"Loading {config['model_1_name']}-{config['model_1_type']} Preprocessor ...")
    model_1, vis_processors_1, txt_processors_1 = load_model_and_preprocess(name=config['model_1_name'], 
                                                                            model_type=config['model_1_type'], 
                                                                            is_eval=True, 
                                                                            device=config['device'])
    model_1_dict = {'model': model_1, 'vis_processors': vis_processors_1, 'txt_processors': txt_processors_1}

    model_2, vis_processors_2, txt_processors_2 = load_model_and_preprocess(name=config['model_2_name'], 
                                                                            model_type=config['model_2_type'], 
                                                                            is_eval=True, 
                                                                            device=config['device'])
    model_2_dict = {'model': model_2, 'vis_processors': vis_processors_2, 'txt_processors': txt_processors_2}
    return model_1_dict, model_2_dict

def run_train(args):
    print(f"RUN TRAIN")
    config_path = args.config_path
    config_name = config_path.split('/')[-1][:-4]
    dataset_name = 'llqa'
    
    config = ut.load_config(config_path)
    config['util_norm'] = False
    config['dataset_name'] = dataset_name
    config['config_path'] = config_path
    config['out_dir'] = f"{config['out_dir']}/{dataset_name}"
    
    model_1_dict, model_2_dict = get_model_dict(config)
    
    train_dataset = ut.Retrieval_Dataset(config=config, list_images=LL_TRAIN_IMAGES, list_texts=LL_TRAIN_TEXTS, 
                                         list_groups=LL_TRAIN_GROUP,
                                         model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                         type_dataset='train')
    val_dataset = ut.Retrieval_Dataset(config=config, list_images=LL_VAL_IMAGES, list_texts=LL_VAL_TEXTS,
                                       list_groups=LL_VAL_GROUP,
                                       model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                       type_dataset='val')
    
    niters = int(int(np.ceil(len(train_dataset) / config['batch_size'])))
    
    if config['Tmax'] > 0:
        config['Tmax'] = config['Tmax'] * niters
    
    if "both" in config_path.lower():
        print("Using weight of 2 backbones")
        controller = Ctr_Both(config)
    else:
        print("Using weight of the dominant backbone")
        controller = Ctr(config)
    
    total_para = controller.count_parameters()
    print(f"Trainable Paras: {total_para}")
    controller.train(dataset_train=train_dataset,  dataset_val=val_dataset,
                     num_epoch=config['num_epoch'], model_name=config_name,
                     groups_dict_it2=VAL_IDX_I2T_DICT, groups_dict_t2i=VAL_IDX_T2I_DICT)
    

def run_evaluate(args):
    config_path = args.config_path
    config_name = config_path.split('/')[-1][:-4]
    dataset_name = 'llqa'
    
    print(f"PERFORM EVALUATE")
    config = ut.load_config(config_path)
    config['out_dir'] = f"{config['out_dir']}/{dataset_name}"    
    save_path = f"{config['out_dir']}/{config_name}/best.pth.tar"
    config['config_path'] = config_path
    config['util_norm'] = False
    config['dataset_name'] = dataset_name
    model_1_dict, model_2_dict = get_model_dict(config)
    
    test_dataset = ut.Retrieval_Dataset(config=config, list_images=LL_VAL_IMAGES, list_texts=LL_VAL_TEXTS,
                                        list_groups=LL_VAL_GROUP,
                                        model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                        type_dataset='test')
    
    controller = Ctr(config)
        
    controller.load_model(save_path)
    controller.eval_mode()
    
    apply_temp = True if controller.temp > 0 else False
    with torch.no_grad():
        r, loss_rall = controller.evaluate_multimodal(test_dataset, apply_temp, return_sim=False,
                                                      groups_dict_i2t=VAL_IDX_I2T_DICT, groups_dict_t2i=VAL_IDX_T2I_DICT)
        r1i, r5i, r10i, r1t, r5t, r10t = r
        
    info_txt = f"R1i: {r1i}\nR5i: {r5i}\nR10i: {r10i}\n"
    info_txt += f"R1t: {r1t}\nR5t: {r5t}\nR10t: {r10t}\n"
    info_txt += f"Ri: {r1i+r5i+r10i}\nRt: {r1t+r5t+r10t}\n"
    info_txt += f"Rall: {r1i+r5i+r10i+r1t+r5t+r10t}\n"
    info_txt += f"LALL: {controller.weight_nll_loss*loss_rall['nll'] + controller.weight_itm_loss*loss_rall['itm']}\n"
    info_txt += f"LNLL: {loss_rall['nll']}\n"
    info_txt += f"LITM: {loss_rall['itm']}\n"
    info_txt += f"LoRe: {loss_rall['r']}\n"
    write_to_file(f"{config['out_dir']}/{config_name}/TestReport.log", info_txt)     
    print(info_txt)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='HADA_m/Config/C5.yml', help='yml file of the config')
    # parser.add_argument('-md', '--model_type', type=str, default='LiFu_m', help='structure of the model')
    parser.add_argument('-rm', '--run_mode', type=str, default='train', help='train: train and test\ntest: only test')
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    print(f"CONFIG: {CONFIG_PATH.split('/')[-1]}")
    if args.run_mode == 'train':
        run_train(args)
        run_evaluate(args)
    if args.run_mode == 'test':
        run_evaluate(args)