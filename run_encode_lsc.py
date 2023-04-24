from Utils import Retrieval_Dataset, load_config, make_dataloader
from lavis.models import load_model_and_preprocess, load_model, load_preprocess
from Controller_Both import Controller as Ctr_Both
from Controller import Controller as Ctr
import xml.etree.ElementTree as ET
import os
from PIL import Image
import pandas as pd
import json
import numpy as np
import argparse

IMG_DIR = '/mnt/data/mount_4TBSSD/nmduy/webp_images'
photo_ids = pd.read_csv(f"/mnt/data/mount_4TBSSD/nmduy/photo_ids.csv")
LIST_IMAGES = photo_ids['photo_id'].tolist()
LIST_PATH_IMAGES = [f"{IMG_DIR}/{x.split('_')[-2][:4]}-{x.split('_')[-2][4:6]}-{x.split('_')[-2][6:]}/{x}.webp" if len(x) > 19 
                    else f"{IMG_DIR}/{x.split('_')[0][:4]}-{x.split('_')[0][4:6]}-{x.split('_')[0][6:]}/{x}.webp"
                    for x in LIST_IMAGES]
# LIST_PATH_IMAGES = LIST_PATH_IMAGES[:10] # DEBUG
lsc21_query_0 = ['building a computer',
                'going into NorthSide Shopping Centre',
                'birds in a cage',
                'a white t-shirt for sale',
                'Planning a thesis/dissertation on a whiteboard with a student',
                'roasting marshmallows',
                'getting too much junk mail',
                'shopping for blue cups',
                'lost and looking for directions',
                'drinking coffee while waiting in a car repair / sales store',
                'Telescope in the mirror',
                'buy a blood pressure monitor',
                'an orange suitcase',
                'TagHeuer advertisement for a watch',
                'Boarding pass for PVG',
                'a colleague in my office carrying a large paper envelope full of documents',
                'taking a photo of a phone screen',
                'looking at small computer chips on rolls',
                'buying fruit from a convenience store',
                'buying glenisk yoghurt',
                'Eating mandarins',
                'learning to fix the broken key of macbook air',
                'playing a retro car-racing game on a laptop in my office'
]

lsc21_query_30 = ['building a computer at desk',
                'going into NorthSide Shopping Centre to get new keys',
                'birds in a cage with a yellow bird at the lower left',
                'a white t-shirt for sale saying I love bicycle',
                'Planning a thesis/dissertation on a whiteboard with a student wearing a blue and black stripey top',
                'roasting marshmallows on a BBQ',
                'put a sign on my door as i was getting too much junk mail',
                'shopping for blue cups with someone wearing a blue jacket',
                'lost and looking for directions on a street',
                'drinking coffee while waiting in a car repair / sales store called Joe Duffy',
                'seeing a telescope and a red flower vase in the mirror in the bedroom',
                'looking to buy a blood pressure monitor in a pharmacy',
                'an orange ride-on suitcase',
                'TagHeuer advertisement for a watch showing a footballer and a watch',
                'queuing at the airport gate with boarding pass for PVG',
                'a colleague in my office carrying a large heavy looking paper envelope full of documents',
                'taking a photo of a phone screen being held by a lady',
                'looking at small computer chips on rolls in a small university electronics laboratory',
                'buying fruit in a convenience store for €2.99',
                'buying glenisk yoghurt in a supermarket',
                'Eating mandarins and an apple',
                'learning to fix the broken key of macbook air by watching a video about it on a monitor',
                'playing a retro single car-racing game at beginner level on a laptop in my office'
]

lsc21_query_60 = ['building a computer at desk with a blue background',
                'going into NorthSide Shopping Centre to get new keys',
                'birds in a cage with a yellow bird on the lower left and a box with a small green beetle-like car',
                'a white t-shirt for sale saying I love bicycle in a bicycle and parts store',
                'Planning a thesis/dissertation on a whiteboard with a student wearing a blue and black stripey top in my office',
                'roasting marshmallows on a BBQ at home',
                'put a sign on my door asking for no more junk mail',
                'shopping for blue cups with someone wearing a blue jacket and bought two bags full of stuff',
                'lost and looking for directions on a street close to an asian restaurant called Maple Leaf',
                'drinking coffee while waiting in a car repair / sales store called Joe Duffy which sold both Volvo and Mazda cars',
                'seeing a telescope and a red flower vase in the mirror in the bedroom. There is a white violin too',
                'looking to buy a blood pressure monitor in a pharmacy that sold Omron and Braun devices',
                'an orange ride-on suitcase with a face',
                'TagHeuer advertisement for a watch showing a footballer sideways kicking the ball and a watch',
                'queuing at the airport gate with boarding pass for PVG on a day with nice blue sky',
                'a colleague wearing red trousers in my office carrying a large heavy looking paper envelope full of documents',
                'taking a photo of a phone screen being held by a lady carrying a suitcase',
                'looking at small computer chips on rolls in a small university electronics laboratory which had at least 100 rolls',
                'buying fruit in a convenience store for €2.99 after a work break from office',
                'buying glenisk yoghurt in a SPAR supermarket',
                'Eating mandarins and an apple while working on a paper',
                'learning to fix the broken key of macbook air by watching a video about it on a monitor. The macbook air was beside the monitor',
                'playing a retro single car-racing game at beginner level on a laptop in my office for a few minutes'
]

num_queries = len(lsc21_query_0)
LIST_TEXTS = lsc21_query_0 + lsc21_query_30 + lsc21_query_60

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

def run_encode_unidata(ctr, dataset, unidata='img'):
    dataset.set_branch(branch=unidata)
    dataloader = make_dataloader(dataset, branch=unidata, 
                                 batch_size=int(ctr.config['batch_size']), 
                                 shuffle=False)
    enc, enc_ori_1, enc_ori_2 = ctr.eval_encode(dataloader, branch=unidata)
    enc = enc.cpu().numpy()
    enc_ori_1 = enc_ori_1.cpu().numpy()
    enc_ori_2 = enc_ori_2.cpu().numpy()
    np.save(f'hada_{unidata}_ft.npy', enc)
    np.save(f'blip_{unidata}_ft.npy', enc_ori_1)
    np.save(f'clip_{unidata}_ft.npy', enc_ori_2)
    
def run_main(args):
    config_path = args.config_path
    run_mode = args.run_mode
    config_name = config_path.split('/')[-1][:-4]
    dataset_name = 'llqa'
    print(f"PERFORM EVALUATE")
    config = load_config(config_path)
    config['out_dir'] = f"{config['out_dir']}/{dataset_name}"    
    save_path = f"{config['out_dir']}/{config_name}/best.pth.tar"
    # save_path = config['pretrained_path']
    config['config_path'] = config_path
    config['util_norm'] = False
    config['dataset_name'] = dataset_name
    model_1_dict, model_2_dict = get_model_dict(config)

    if "both" in config_path.lower():
        print("Using weight of 2 backbones")
        controller = Ctr_Both(config)
    else:
        print("Using weight of the dominant backbone")
        controller = Ctr(config)
        
    controller.load_model(save_path)
    controller.eval_mode()

    dataset = Retrieval_Dataset(config=config, list_images=LIST_PATH_IMAGES, list_texts=LIST_TEXTS, 
                                list_groups=None,
                                model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                type_dataset='test')
    
    if run_mode in ['img', 'image', 'imgs', 'images', 'both']:
        print('ENCODING IMAGES ...')
        run_encode_unidata(controller, dataset, unidata='img')
    if run_mode in ['txt', 'text', 'texts', 'caption', 'captions', 'both']:
        print('ENCODING TEXTS ...')
        run_encode_unidata(controller, dataset, unidata='txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='HADA_m/Config/C5.yml', help='yml file of the config')
    # parser.add_argument('-md', '--model_type', type=str, default='LiFu_m', help='structure of the model')
    parser.add_argument('-rm', '--run_mode', type=str, default='image', help='image, text, both')
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    print(f"CONFIG: {CONFIG_PATH.split('/')[-1]}")
    run_main(args)