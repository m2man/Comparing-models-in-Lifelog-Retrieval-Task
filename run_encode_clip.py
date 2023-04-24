from lavis.models import load_model_and_preprocess, load_model, load_preprocess
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
import pandas as pd
import json
import numpy as np
import torch
from tqdm import tqdm

# model_1_name: blip_retrieval
# model_1_type: flickr
# model_2_name: clip_feature_extractor
# model_2_type: ViT-L-14-336

model_name = "clip_feature_extractor"
model_type = "ViT-B-16" #"ViT-L-14-336"
is_clip = "clip" in model_name
device = "cuda"

IMG_DIR = '/mnt/data/mount_4TBSSD/nmduy/webp_images'

lsc21_query_0 = ['buiding a computer',
'going into NorthSide Shopping Centre',
'birds in a cage',
'a while t-shirt for sale',
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

lsc21_query_30 = ['buiding a computer at desk',
'going into NorthSide Shopping Centre to get new keys',
'birds in a cage with a yellow bird at the lower left',
'a while t-shirt for sale saying I love bicycle',
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

lsc21_query_60 = [
'buiding a computer at desk with a blue background',
'going into NorthSide Shopping Centre to get new keys',
'birds in a cage with a yellow bird on the lower left and a box with a small green beetle-like car',
'a while t-shirt for sale saying I love bicycle in a bicycle and parts store',
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
'looking at small computer chips on rolls in a small university electronics laboratory which had atleat 100 rolls',
'buying fruit in a convenience store for €2.99 after a work break from office',
'buying glenisk yoghurt in a SPAR supermarket',
'Eating mandarins and an apple while working on a paper',
'learning to fix the broken key of macbook air by watching a video about it on a monitor. The macbook air was beside the monitor',
'playing a retro single car-racing game at beginner level on a laptop in my office for a few minutes'
]

def get_model_dict(model_name, model_type, device="cuda"):
    print(f"Loading {model_name}-{model_type} Preprocessor ...")
    model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, 
                                                                      model_type=model_type, 
                                                                      is_eval=True, 
                                                                      device=device)
    model_dict = {'model': model, 'vis_processors': vis_processors, 'txt_processors': txt_processors}
    return model_dict

def encode_image(image_path, model, vis_processors, is_clip=False):
    image_raw = Image.open(image_path)
    image_input = vis_processors["eval"](image_raw).unsqueeze(0).to(model.device)
    if is_clip:
        image_features = model.encode_image(image_input)
    else:
        image_embeds = model.visual_encoder.forward_features(image_input)
        image_features = model.vision_proj(image_embeds)
    image_features = F.normalize(image_features, dim=-1)
    return image_features

def encode_text(text, model, txt_processors, is_clip=False):
    text_input = txt_processors["eval"](text)
    if is_clip:
        text_token = model.tokenizer(text_input).to(model.device)
        text_features = model.encode_text(text_token)
    else:
        text_token = model.tokenizer(text_input, return_tensors="pt", padding=True).to(model.device)
        text_output = model.text_encoder(
            text_token.input_ids,
            attention_mask=text_token.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        text_features = model.text_proj(text_embeds)
    text_features = F.normalize(text_features, dim=-1)
    return text_features

photo_ids = pd.read_csv(f"/mnt/data/mount_4TBSSD/nmduy/photo_ids.csv")
LIST_IMAGES = photo_ids['photo_id'].tolist()
LIST_PATH_IMAGES = [f"{IMG_DIR}/{x.split('_')[-2][:4]}-{x.split('_')[-2][4:6]}-{x.split('_')[-2][6:]}/{x}.webp" if len(x) > 19 
                    else f"{IMG_DIR}/{x.split('_')[0][:4]}-{x.split('_')[0][4:6]}-{x.split('_')[0][6:]}/{x}.webp"
                    for x in LIST_IMAGES]
LIST_TEXTS = lsc21_query_0 + lsc21_query_30 + lsc21_query_60

model_dict = get_model_dict(model_name, model_type, device="cuda")

enc_image = torch.tensor(())
for image_path in tqdm(LIST_PATH_IMAGES):
    x = encode_image(image_path, model_dict['model'], 
                     model_dict['vis_processors'], is_clip)
    x = x.detach().cpu()
    enc_image = torch.cat((enc_image, x))
enc_image = enc_image.numpy()
np.save(f'clip_img_ft.npy', enc_image)

enc_text = torch.tensor(())
for text in tqdm(LIST_TEXTS):
    x = encode_text(text, model_dict['model'], 
                     model_dict['txt_processors'], is_clip)
    x = x.detach().cpu()
    enc_text = torch.cat((enc_text, x))
enc_text = enc_text.numpy()
np.save(f'clip_txt_ft.npy', enc_text)