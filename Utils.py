import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import random
import torch
import itertools
import yaml
import torch.nn.functional as F
from RandAugment import RandomAugment
from PIL import Image
import torchvision.transforms as transforms
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from lavis.models import load_model_and_preprocess, load_model, load_preprocess
# from lavis.datasets.builders import load_dataset
# flickr_dataset = load_dataset("flickr30k",vis_path='/mnt/data/itr_dataset/dataset/flickr30k_images/')


clip_train_transform = transforms.Compose([                        
        transforms.RandomResizedCrop(336, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                          'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToPILImage(),
        # transforms.ToTensor(),
        # normalize,
    ])  
    
# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    
def do_normalize(data, para_min, para_max):
    normed = (data - para_min) / (para_max - para_min)
    return normed

def do_standardize(data, mean, std):
    normed = (data - mean) / std
    return normed

def load_config(filename):
    with open(filename) as file:
        config_dict= yaml.safe_load(file)
    return config_dict

def write_to_file(filepath='text.txt', content=''):
    with open(filepath, "a") as f_log:
        f_log.write(content)

def create_index_from_2_list(list_1, list_2, dual_index=False, self_loop=False):
    first = np.repeat(list_1, len(list_2))
    second = np.tile(list_2, len(list_1))
    result = np.asarray([first, second])
    if dual_index:
        first = np.repeat(list_2, len(list_1))
        second = np.tile(list_1, len(list_2))
        result = np.concatenate((result, np.asarray([first, second])), axis=1)
    if self_loop:
        list_all = list_1 + list_2
        result = np.concatenate((result, np.asarray([list_all, list_all])), axis=1)
    return result

def create_index(start_idx, end_idx, self_loop=True):
    # create index (2, (end_idx-start_idx)**2) 
    n = end_idx - start_idx + 1
    r = [x for x in range(start_idx, end_idx+1)]
    first = np.repeat(r, n)
    second = np.tile(r, n)
    result = np.asarray([first, second])
    if not self_loop:
        result = result[:,np.where((result[0] - result[1]) != 0)[0]]
    return result

def create_edge_index(list_num_nodes, self_loop=False):
    batch_size = len(list_num_nodes)
    count = 0
    for idx in range(batch_size):
        this_num_nodes = list_num_nodes[idx]
        if idx == 0:
            edge_indices = create_index(count, count+this_num_nodes-1, self_loop=self_loop)
        else:
            edge_indices = np.concatenate((edge_indices,
                                           create_index(count, count+this_num_nodes-1, self_loop=self_loop)),
                                           axis=1)
        count += this_num_nodes 
    edge_indices = torch.tensor(edge_indices) # (2, total edges)                                       
    return edge_indices

def extract_features(model, samples, mode='image', is_clip=False):
    # mode = 'image' or 'text'
    model.eval()
    image = samples.get("image")
    caption = samples.get("text_input")
    with torch.no_grad():
        if mode == 'image':
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            if is_clip:
                image_embeds = model.encode_image(image)
                image_features = torch.clone(image_embeds)
                # image_features = F.normalize(image_embeds, dim=-1)
            else:
                image_embeds = model.visual_encoder.forward_features(image)
                image_features = model.vision_proj(image_embeds)
                # image_features = F.normalize(image_features, dim=-1)
            text_embeds = None
            text_features = None
        if mode == 'text':
            assert (
                caption is not None
            ), "text input is None for mode 'text'"
            if is_clip:
                text = model.tokenizer(caption).to(model.device)
                text_embeds = model.encode_text(text)
                text_features = torch.clone(text_embeds)
                # text_features = F.normalize(text_embeds, dim=-1)
            else:
                text = model.tokenizer(caption, return_tensors="pt", padding=True).to(
                    model.device
                )
                text_output = model.text_encoder(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                text_features = model.text_proj(text_embeds)
                # text_features = F.normalize(text_features, dim=-1)
            image_embeds = None
            image_features = None
    # proj need to F.normalize(dim=-1) to perform retrieval
    return {'image_embeds': image_embeds,
            'image_embeds_proj': image_features,
            'text_embeds': text_embeds,
            'text_embeds_proj': text_features}

class Retrieval_Dataset(Dataset):
    def __init__(self, config, list_images, list_texts, list_groups, model_1_dict, model_2_dict, type_dataset='train'):  
        assert type_dataset in ['train', 'val', 'test'], f"Not supported type_dataset: {type_dataset}"
        self.model_1_name = config['model_1_name']
        self.model_1_type = config['model_1_type']
        self.model_2_name = config['model_2_name']
        self.model_2_type = config['model_2_type']
        self.device = torch.device(config['device'])
        self.directed_graph = config['directed_graph']
        self.type_dataset = type_dataset
        self.wfc = config['wfc']
        self.wfcx = config['wfcx']
        self.wcc = config['wcc']
        self.wfsl = config['wfsl']
        self.wcsl = config['wcsl']
        self.model_1 = model_1_dict['model']
        self.vis_processors_1 = model_1_dict['vis_processors']
        self.txt_processors_1 = model_1_dict['txt_processors']
        self.model_2 = model_2_dict['model']
        self.vis_processors_2 = model_2_dict['vis_processors']
        self.txt_processors_2 = model_2_dict['txt_processors']
        self.ori_dataset = type_dataset
        if type_dataset != 'train':
            self.type_dataset = 'eval'
            self.ori_type_dataset = type_dataset
        else:
            self.list_images = list_images
            self.list_texts = list_texts      
        self.ori_list_images = list_images
        self.ori_list_texts = list_texts
        self.list_groups = list_groups
        self.ori_unique_list_images = list(set(list_images))
        self.ori_unique_list_texts = list(set(list_texts))
    
    def set_branch(self, branch='img'):
        # brach == 'img', 'image', 'text', 'txt', 'cap'
        if self.type_dataset == 'train':
            print("type_dataset is train --> no need to run this set_branch function ...")
            pass
        else:
            print(f"Set dataset branch to [{branch}]")
            self.branch = branch
            if self.branch in ['img', 'image']:
                self.list_images = self.ori_list_images
                self.list_texts = None
            else: # text -- txt -- cap
                self.list_texts = self.ori_list_texts
                self.list_images = None
    
    def set_branch_unique(self, branch='img'):
        if self.type_dataset == 'train':
            print("type_dataset is train --> no need to run this set_branch_unique function ...")
            pass
        else:
            print(f"Set dataset UNIQUE branch to [{branch}]")
            self.branch = branch
            if self.branch in ['img', 'image']:
                self.list_images = self.ori_unique_list_images
                self.list_texts = None
            else: # text -- txt -- cap
                self.list_texts = self.ori_unique_list_texts
                self.list_images = None
        return None
                
    def __len__(self):
        if self.list_images is not None:
            return len(self.list_images)
        else:
            return len(self.list_texts)
        
    def __getitem__(self, index): 
        dict_return = {}
        if self.list_groups is not None:
            dict_return['image_id'] = torch.tensor([int(self.list_groups[index])])
            dict_return['instance_id'] = torch.tensor([int(self.list_groups[index])])
        else:
            dict_return['image_id'] = torch.tensor([int(index)])
            dict_return['instance_id'] = torch.tensor([int(index)])
        if self.list_images is not None:
            image_raw = self.list_images[index]
            image_raw = Image.open(image_raw)
        else:
            image_raw = None
        if self.list_texts is not None:
            text_raw = self.list_texts[index]
        else:
            text_raw = None
        # preprocess data 1
        if "base" in self.model_1_type:
            image_input_1 = self.vis_processors_1["eval"](image_raw).unsqueeze(0).to(self.device) if image_raw is not None else None
            text_input_1 = self.txt_processors_1["eval"](text_raw) if text_raw is not None else None
        else:
            image_input_1 = self.vis_processors_1[self.type_dataset](image_raw).unsqueeze(0).to(self.device) if image_raw is not None else None
            text_input_1 = self.txt_processors_1[self.type_dataset](text_raw) if text_raw is not None else None
        # preprocess data 2
        if 'clip' in self.model_2_name and self.type_dataset == 'train':
            # clip_image_raw = clip_train_transform(image_raw) if image_raw is not None else None
            image_input_2 = self.vis_processors_2["eval"](image_raw).unsqueeze(0).to(self.device) if image_raw is not None else None
            text_input_2 = self.txt_processors_2["eval"](text_raw) if text_raw is not None else None
        else:
            image_input_2 = self.vis_processors_2[self.type_dataset](image_raw).unsqueeze(0).to(self.device) if image_raw is not None else None
            text_input_2 = self.txt_processors_2[self.type_dataset](text_raw) if text_raw is not None else None
        sample_1 = {"image": image_input_1, "text_input": text_input_1}
        sample_2 = {"image": image_input_2, "text_input": text_input_2}
        
        if image_raw is not None:
            image_output_1 = extract_features(self.model_1, sample_1, mode='image', is_clip='clip' in self.model_1_name)
            image_output_2 = extract_features(self.model_2, sample_2, mode='image', is_clip='clip' in self.model_2_name)
            img_ft_1 = image_output_1['image_embeds'].squeeze()
            img_ft_proj_1 = image_output_1['image_embeds_proj'][:,0,:]
            img_ft_2 = image_output_2['image_embeds']
            img_ft_proj_2 = image_output_2['image_embeds_proj']
            img_ft_proj_1_ori = F.normalize(img_ft_proj_1, dim=-1)
            img_ft_proj_2_ori = F.normalize(img_ft_proj_2, dim=-1)
            # IMAGE GRAPH
            n_1 = img_ft_1.shape[0]
            n_2 = img_ft_2.shape[0]
            index_1 = [x for x in range(n_1)]
            index_2 = [n_1+x for x in range(n_2)]
            # create edge here
            c1 = [index_1[0]]
            c2 = [index_2[0]]
            f1 = index_1[1:]
            f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=False) # (2, n_edge)
            f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
            c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
            f1_sl = create_index_from_2_list(f1, [], dual_index=False, self_loop=True)
            c1_sl = create_index_from_2_list(c1, [], dual_index=False, self_loop=True)
            c2_sl = create_index_from_2_list(c2, [], dual_index=False, self_loop=True)
            edge_index = np.concatenate((f1_c1, f1_c2, c1_c2, f1_sl, c1_sl, c2_sl), axis=1)
            # create edge attr
            f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
            f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
            c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
            f1_sl_attr = [self.wfsl for x in range(f1_sl.shape[1])]
            c1_sl_attr = [self.wcsl for x in range(c1_sl.shape[1])]
            c2_sl_attr = [self.wcsl for x in range(c2_sl.shape[1])]
            edge_attr = np.array(f1_c1_attr + f1_c2_attr + c1_c2_attr + f1_sl_attr + c1_sl_attr + c2_sl_attr)
            img_edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            img_edge_index = torch.tensor(edge_index, dtype=torch.int64)
            dict_return['img'] = {'ft_1': img_ft_1,
                                  'ft_proj_1': img_ft_proj_1,
                                  'ft_2': img_ft_2,
                                  'ft_proj_2': img_ft_proj_2,
                                  'edge_attr': img_edge_attr,
                                  'edge_index': img_edge_index,
                                  'ft_proj_ori_1': img_ft_proj_1_ori, 
                                  'ft_proj_ori_2': img_ft_proj_2_ori} 
            
        if text_raw is not None:
            text_output_1 = extract_features(self.model_1, sample_1, mode='text', is_clip='clip' in self.model_1_name) 
            text_output_2 = extract_features(self.model_2, sample_2, mode='text', is_clip='clip' in self.model_2_name)
            txt_ft_1 = text_output_1['text_embeds'].squeeze()
            txt_ft_proj_1 = text_output_1['text_embeds_proj'][:,0,:]
            txt_ft_2 = text_output_2['text_embeds']
            txt_ft_proj_2 = text_output_2['text_embeds_proj']
            txt_ft_proj_1_ori = F.normalize(txt_ft_proj_1, dim=-1)
            txt_ft_proj_2_ori = F.normalize(txt_ft_proj_2, dim=-1)
            # TXT GRAPH
            n_1 = txt_ft_1.shape[0]
            n_2 = txt_ft_2.shape[0]
            index_1 = [x for x in range(n_1)]
            index_2 = [n_1+x for x in range(n_2)]
            # create edge here
            c1 = [index_1[0]]
            c2 = [index_2[0]]
            f1 = index_1[1:]
            f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=False) # (2, n_edge)
            f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
            c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
            f1_sl = create_index_from_2_list(f1, [], dual_index=False, self_loop=True)
            c1_sl = create_index_from_2_list(c1, [], dual_index=False, self_loop=True)
            c2_sl = create_index_from_2_list(c2, [], dual_index=False, self_loop=True)
            edge_index = np.concatenate((f1_c1, f1_c2, c1_c2, f1_sl, c1_sl, c2_sl), axis=1)
            # create edge attr
            f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
            f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
            c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
            f1_sl_attr = [self.wfsl for x in range(f1_sl.shape[1])]
            c1_sl_attr = [self.wcsl for x in range(c1_sl.shape[1])]
            c2_sl_attr = [self.wcsl for x in range(c2_sl.shape[1])]
            edge_attr = np.array(f1_c1_attr + f1_c2_attr + c1_c2_attr + f1_sl_attr + c1_sl_attr + c2_sl_attr)
            txt_edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            txt_edge_index = torch.tensor(edge_index, dtype=torch.int64)
            dict_return['cap'] = {'ft_1': txt_ft_1,
                                  'ft_proj_1': txt_ft_proj_1,
                                  'ft_2': txt_ft_2,
                                  'ft_proj_2': txt_ft_proj_2,
                                  'edge_attr': txt_edge_attr,
                                  'edge_index': txt_edge_index,
                                  'ft_proj_ori_1': txt_ft_proj_1_ori, 
                                  'ft_proj_ori_2': txt_ft_proj_2_ori}
        return dict_return

    
def collate_function_both(batch):
    ts_cap_ft_1 = torch.tensor(()).to(batch[0]['cap']['ft_1'].device)
    ts_cap_ft_2 = torch.tensor(()).to(batch[0]['cap']['ft_2'].device)
    ts_cap_ft_proj_1 = torch.tensor(()).to(batch[0]['cap']['ft_1'].device)
    ts_cap_ft_proj_2 = torch.tensor(()).to(batch[0]['cap']['ft_2'].device)
    ts_cap_ft_proj_ori_1 = torch.tensor(()).to(batch[0]['cap']['ft_1'].device)
    ts_cap_ft_proj_ori_2 = torch.tensor(()).to(batch[0]['cap']['ft_2'].device)
    list_cap_edge_index = []
    list_cap_edge_attr = []
    list_n_cap_node = []
    list_n_cap_node_1 = []
    list_n_cap_node_2 = []
    
    ts_img_ft_1 = torch.tensor(()).to(batch[0]['img']['ft_1'].device)
    ts_img_ft_2 = torch.tensor(()).to(batch[0]['img']['ft_2'].device)
    ts_img_ft_proj_1 = torch.tensor(()).to(batch[0]['img']['ft_1'].device)
    ts_img_ft_proj_2 = torch.tensor(()).to(batch[0]['img']['ft_2'].device)
    ts_img_ft_proj_ori_1 = torch.tensor(()).to(batch[0]['img']['ft_1'].device)
    ts_img_ft_proj_ori_2 = torch.tensor(()).to(batch[0]['img']['ft_2'].device)
    list_img_edge_index = []
    list_img_edge_attr = []
    list_n_img_node = []
    list_n_img_node_1 = []
    list_n_img_node_2 = []
    
    ts_image_id = torch.tensor(())
    ts_instance_id = torch.tensor(())
    
    for x in batch:
        image_id = x['image_id']
        instance_id = x['instance_id']
        ts_image_id = torch.cat((ts_image_id, image_id)) # (bs,)
        ts_instance_id = torch.cat((ts_instance_id, instance_id)) #(bs,)
        
        ts_cap_ft_1 = torch.cat((ts_cap_ft_1,x['cap']['ft_1']))
        ts_cap_ft_2 = torch.cat((ts_cap_ft_2,x['cap']['ft_2']))
        ts_cap_ft_proj_1 = torch.cat((ts_cap_ft_proj_1,x['cap']['ft_proj_1']))
        ts_cap_ft_proj_2 = torch.cat((ts_cap_ft_proj_2,x['cap']['ft_proj_2']))
        ts_cap_ft_proj_ori_1 = torch.cat((ts_cap_ft_proj_ori_1,x['cap']['ft_proj_ori_1']))
        ts_cap_ft_proj_ori_2 = torch.cat((ts_cap_ft_proj_ori_2,x['cap']['ft_proj_ori_2']))
        
        list_cap_edge_index.append(x['cap']['edge_index'])
        list_cap_edge_attr.append(x['cap']['edge_attr'])
        list_n_cap_node.append(x['cap']['ft_1'].shape[0] + x['cap']['ft_2'].shape[0])
        list_n_cap_node_1.append(x['cap']['ft_1'].shape[0])
        list_n_cap_node_2.append(x['cap']['ft_2'].shape[0])
        
        ts_img_ft_1 = torch.cat((ts_img_ft_1,x['img']['ft_1']))
        ts_img_ft_2 = torch.cat((ts_img_ft_2,x['img']['ft_2']))
        ts_img_ft_proj_1 = torch.cat((ts_img_ft_proj_1,x['img']['ft_proj_1']))
        ts_img_ft_proj_2 = torch.cat((ts_img_ft_proj_2,x['img']['ft_proj_2']))
        ts_img_ft_proj_ori_1 = torch.cat((ts_img_ft_proj_ori_1,x['img']['ft_proj_ori_1']))
        ts_img_ft_proj_ori_2 = torch.cat((ts_img_ft_proj_ori_2,x['img']['ft_proj_ori_2']))
        
        list_img_edge_index.append(x['img']['edge_index'])
        list_img_edge_attr.append(x['img']['edge_attr'])
        list_n_img_node.append(x['img']['ft_1'].shape[0] + x['img']['ft_2'].shape[0])
        list_n_img_node_1.append(x['img']['ft_1'].shape[0])
        list_n_img_node_2.append(x['img']['ft_2'].shape[0])
        
    ts_image_id = ts_image_id.reshape(-1,1)  
    ts_instance_id = ts_instance_id.reshape(-1,1)
    
    bs = len(ts_image_id)
    img_edge_attr = torch.cat(list_img_edge_attr)
    cap_edge_attr = torch.cat(list_cap_edge_attr)
    del list_img_edge_attr, list_cap_edge_attr
    
    img_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_img_node))
    cap_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_cap_node))
    
    count_img = 0
    count_cap = 0
    for idx in range(bs):
        list_img_edge_index[idx] = list_img_edge_index[idx] + count_img
        list_cap_edge_index[idx] = list_cap_edge_index[idx] + count_cap
        count_img += list_n_img_node[idx]
        count_cap += list_n_cap_node[idx]
    img_edge_index = torch.cat(list_img_edge_index, dim=1)
    cap_edge_index = torch.cat(list_cap_edge_index, dim=1)
    del list_img_edge_index, list_cap_edge_index
    
    n_img_node_1 = torch.tensor(list_n_img_node_1)
    n_img_node_2 = torch.tensor(list_n_img_node_2)
    n_cap_node_1 = torch.tensor(list_n_cap_node_1)
    n_cap_node_2 = torch.tensor(list_n_cap_node_2)
    del list_n_img_node_1, list_n_img_node_2, list_n_cap_node_1, list_n_cap_node_2
    
    img_dict = {'ft_proj_1': ts_img_ft_proj_1, 'ft_proj_2': ts_img_ft_proj_2,
                'ft_1': ts_img_ft_1, 'ft_2': ts_img_ft_2, 'batch_index': img_batch_index,
                'ft_proj_ori_1': ts_img_ft_proj_ori_1, 'ft_proj_ori_2': ts_img_ft_proj_ori_2,
                'n_node_1': n_img_node_1, 'n_node_2': n_img_node_2,
                'edge_index': img_edge_index, 'edge_attr': img_edge_attr}
    
    cap_dict = {'ft_proj_1': ts_cap_ft_proj_1, 'ft_proj_2': ts_cap_ft_proj_2,
                'ft_1': ts_cap_ft_1, 'ft_2': ts_cap_ft_2, 'batch_index': cap_batch_index,
                'ft_proj_ori_1': ts_cap_ft_proj_ori_1, 'ft_proj_ori_2': ts_cap_ft_proj_ori_2,
                'n_node_1': n_cap_node_1, 'n_node_2': n_cap_node_2,
                'edge_index': cap_edge_index, 'edge_attr': cap_edge_attr}
    
    return img_dict, cap_dict, ts_image_id, ts_instance_id

def collate_function_img(batch):
    ts_img_ft_1 = torch.tensor(()).to(batch[0]['img']['ft_1'].device)
    ts_img_ft_2 = torch.tensor(()).to(batch[0]['img']['ft_2'].device)
    ts_img_ft_proj_1 = torch.tensor(()).to(batch[0]['img']['ft_1'].device)
    ts_img_ft_proj_2 = torch.tensor(()).to(batch[0]['img']['ft_2'].device)
    ts_img_ft_proj_ori_1 = torch.tensor(()).to(batch[0]['img']['ft_1'].device)
    ts_img_ft_proj_ori_2 = torch.tensor(()).to(batch[0]['img']['ft_2'].device)
    list_img_edge_index = []
    list_img_edge_attr = []
    list_n_img_node = []
    list_n_img_node_1 = []
    list_n_img_node_2 = []
    ts_image_id = torch.tensor(())
    
    for x in batch:        
        image_id = x['image_id']
        ts_image_id = torch.cat((ts_image_id, image_id)) # (bs,)
        
        ts_img_ft_1 = torch.cat((ts_img_ft_1,x['img']['ft_1']))
        ts_img_ft_2 = torch.cat((ts_img_ft_2,x['img']['ft_2']))
        ts_img_ft_proj_1 = torch.cat((ts_img_ft_proj_1,x['img']['ft_proj_1']))
        ts_img_ft_proj_2 = torch.cat((ts_img_ft_proj_2,x['img']['ft_proj_2']))
        ts_img_ft_proj_ori_1 = torch.cat((ts_img_ft_proj_ori_1,x['img']['ft_proj_ori_1']))
        ts_img_ft_proj_ori_2 = torch.cat((ts_img_ft_proj_ori_2,x['img']['ft_proj_ori_2']))
        
        list_img_edge_index.append(x['img']['edge_index'])
        list_img_edge_attr.append(x['img']['edge_attr'])
        list_n_img_node.append(x['img']['ft_1'].shape[0] + x['img']['ft_2'].shape[0])
        list_n_img_node_1.append(x['img']['ft_1'].shape[0])
        list_n_img_node_2.append(x['img']['ft_2'].shape[0])
    
    bs = len(ts_image_id)
    img_edge_attr = torch.cat(list_img_edge_attr)
    del list_img_edge_attr
    img_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_img_node))   
    count_img = 0
    for idx in range(bs):
        list_img_edge_index[idx] = list_img_edge_index[idx] + count_img
        count_img += list_n_img_node[idx]
    img_edge_index = torch.cat(list_img_edge_index, dim=1)
    del list_img_edge_index
    n_img_node_1 = torch.tensor(list_n_img_node_1)
    n_img_node_2 = torch.tensor(list_n_img_node_2)
    del list_n_img_node_1, list_n_img_node_2
    
    img_dict = {'ft_proj_1': ts_img_ft_proj_1, 'ft_proj_2': ts_img_ft_proj_2,
                'ft_1': ts_img_ft_1, 'ft_2': ts_img_ft_2, 'batch_index': img_batch_index,
                'ft_proj_ori_1': ts_img_ft_proj_ori_1, 'ft_proj_ori_2': ts_img_ft_proj_ori_2,
                'n_node_1': n_img_node_1, 'n_node_2': n_img_node_2,
                'edge_index': img_edge_index, 'edge_attr': img_edge_attr}
    
    return img_dict, ts_image_id

def collate_function_cap(batch):
    ts_cap_ft_1 = torch.tensor(()).to(batch[0]['cap']['ft_1'].device)
    ts_cap_ft_2 = torch.tensor(()).to(batch[0]['cap']['ft_2'].device)
    ts_cap_ft_proj_1 = torch.tensor(()).to(batch[0]['cap']['ft_1'].device)
    ts_cap_ft_proj_2 = torch.tensor(()).to(batch[0]['cap']['ft_2'].device)
    ts_cap_ft_proj_ori_1 = torch.tensor(()).to(batch[0]['cap']['ft_1'].device)
    ts_cap_ft_proj_ori_2 = torch.tensor(()).to(batch[0]['cap']['ft_2'].device)
    list_cap_edge_index = []
    list_cap_edge_attr = []
    list_n_cap_node = []
    list_n_cap_node_1 = []
    list_n_cap_node_2 = []
   
    ts_instance_id = torch.tensor(())
    
    for x in batch:
        instance_id = x['instance_id']
        ts_instance_id = torch.cat((ts_instance_id, instance_id)) #(bs,)
        
        ts_cap_ft_1 = torch.cat((ts_cap_ft_1,x['cap']['ft_1']))
        ts_cap_ft_2 = torch.cat((ts_cap_ft_2,x['cap']['ft_2']))
        ts_cap_ft_proj_1 = torch.cat((ts_cap_ft_proj_1,x['cap']['ft_proj_1']))
        ts_cap_ft_proj_2 = torch.cat((ts_cap_ft_proj_2,x['cap']['ft_proj_2']))
        ts_cap_ft_proj_ori_1 = torch.cat((ts_cap_ft_proj_ori_1,x['cap']['ft_proj_ori_1']))
        ts_cap_ft_proj_ori_2 = torch.cat((ts_cap_ft_proj_ori_2,x['cap']['ft_proj_ori_2']))
        
        list_cap_edge_index.append(x['cap']['edge_index'])
        list_cap_edge_attr.append(x['cap']['edge_attr'])
        list_n_cap_node.append(x['cap']['ft_1'].shape[0] + x['cap']['ft_2'].shape[0])
        list_n_cap_node_1.append(x['cap']['ft_1'].shape[0])
        list_n_cap_node_2.append(x['cap']['ft_2'].shape[0])
        
    ts_instance_id = ts_instance_id.reshape(-1,1)
    
    bs = len(ts_instance_id)
    cap_edge_attr = torch.cat(list_cap_edge_attr)
    del list_cap_edge_attr
    cap_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_cap_node))
    count_cap = 0
    for idx in range(bs):
        list_cap_edge_index[idx] = list_cap_edge_index[idx] + count_cap
        count_cap += list_n_cap_node[idx]
    cap_edge_index = torch.cat(list_cap_edge_index, dim=1)
    del list_cap_edge_index
    n_cap_node_1 = torch.tensor(list_n_cap_node_1)
    n_cap_node_2 = torch.tensor(list_n_cap_node_2)
    del list_n_cap_node_1, list_n_cap_node_2
    
    cap_dict = {'ft_proj_1': ts_cap_ft_proj_1, 'ft_proj_2': ts_cap_ft_proj_2,
                'ft_1': ts_cap_ft_1, 'ft_2': ts_cap_ft_2, 'batch_index': cap_batch_index,
                'ft_proj_ori_1': ts_cap_ft_proj_ori_1, 'ft_proj_ori_2': ts_cap_ft_proj_ori_2,
                'n_node_1': n_cap_node_1, 'n_node_2': n_cap_node_2,
                'edge_index': cap_edge_index, 'edge_attr': cap_edge_attr}
    
    return cap_dict, ts_instance_id


def make_dataloader(dataset, branch='both', **args):
    if branch == 'both':
        return DataLoader(dataset, collate_fn=collate_function_both, **args)
    if branch == 'img' or branch == 'image':
        return DataLoader(dataset, collate_fn=collate_function_img, **args)
    if branch == 'cap' or branch == 'txt':
        return DataLoader(dataset, collate_fn=collate_function_cap, **args)

def extract_feature_from_single_text(text, config, model_1_dict, model_2_dict):
    model_1_name = config['model_1_name']
    model_1_type = config['model_1_type']
    model_2_name = config['model_2_name']
    model_2_type = config['model_2_type']
    device = torch.device(config['device'])
    directed_graph = config['directed_graph']
    wfc = config['wfc']
    wfcx = config['wfcx']
    wcc = config['wcc']
    wfsl = config['wfsl']
    wcsl = config['wcsl']
    model_1 = model_1_dict['model']
    txt_processors_1 = model_1_dict['txt_processors']
    model_2 = model_2_dict['model']
    txt_processors_2 = model_2_dict['txt_processors']
    # preprocess data 1
    image_input_1 = None
    text_input_1 = txt_processors_1["eval"](text)
    # preprocess data 2
    image_input_2 = None
    text_input_2 = txt_processors_2["eval"](text)
    sample_1 = {"image": image_input_1, "text_input": text_input_1}
    sample_2 = {"image": image_input_2, "text_input": text_input_2} 
    text_output_1 = extract_features(model_1, sample_1, mode='text', is_clip='clip' in model_1_name) 
    text_output_2 = extract_features(model_2, sample_2, mode='text', is_clip='clip' in model_2_name)
    txt_ft_1 = text_output_1['text_embeds'].squeeze()
    txt_ft_proj_1 = text_output_1['text_embeds_proj'][:,0,:]
    txt_ft_2 = text_output_2['text_embeds']
    txt_ft_proj_2 = text_output_2['text_embeds_proj']
    txt_ft_proj_1_ori = F.normalize(txt_ft_proj_1, dim=-1)
    txt_ft_proj_2_ori = F.normalize(txt_ft_proj_2, dim=-1)
    # TXT GRAPH
    n_1 = txt_ft_1.shape[0]
    n_2 = txt_ft_2.shape[0]
    index_1 = [x for x in range(n_1)]
    index_2 = [n_1+x for x in range(n_2)]
    # create edge here
    c1 = [index_1[0]]
    c2 = [index_2[0]]
    f1 = index_1[1:]
    f1_c1 = create_index_from_2_list(f1, c1, dual_index=not directed_graph, self_loop=False) # (2, n_edge)
    f1_c2 = create_index_from_2_list(f1, c2, dual_index=not directed_graph, self_loop=False)
    c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
    f1_sl = create_index_from_2_list(f1, [], dual_index=False, self_loop=True)
    c1_sl = create_index_from_2_list(c1, [], dual_index=False, self_loop=True)
    c2_sl = create_index_from_2_list(c2, [], dual_index=False, self_loop=True)
    edge_index = np.concatenate((f1_c1, f1_c2, c1_c2, f1_sl, c1_sl, c2_sl), axis=1)
    # create edge attr
    f1_c1_attr = [wfc for x in range(f1_c1.shape[1])]
    f1_c2_attr = [wfcx for x in range(f1_c2.shape[1])]
    c1_c2_attr = [wcc for x in range(c1_c2.shape[1])]
    f1_sl_attr = [wfsl for x in range(f1_sl.shape[1])]
    c1_sl_attr = [wcsl for x in range(c1_sl.shape[1])]
    c2_sl_attr = [wcsl for x in range(c2_sl.shape[1])]
    edge_attr = np.array(f1_c1_attr + f1_c2_attr + c1_c2_attr + f1_sl_attr + c1_sl_attr + c2_sl_attr)
    txt_edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    txt_edge_index = torch.tensor(edge_index, dtype=torch.int64)
    total_node = txt_ft_1.shape[0] + txt_ft_2.shape[0]
    cap_batch_index = torch.tensor([0 for x in range(total_node)])
    n_node_1 = torch.tensor([txt_ft_1.shape[0]])
    n_node_2 = torch.tensor([txt_ft_2.shape[0]])
    cap_dict = {'ft_proj_1': txt_ft_proj_1, 'ft_proj_2': txt_ft_proj_2,
                'ft_1': txt_ft_1, 'ft_2': txt_ft_2, 'batch_index': cap_batch_index,
                'ft_proj_ori_1': txt_ft_proj_1_ori, 'ft_proj_ori_2': txt_ft_proj_2_ori,
                'n_node_1': n_node_1, 'n_node_2': n_node_2,
                'edge_index': txt_edge_index, 'edge_attr': txt_edge_attr}
    
    return cap_dict


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    min_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            min_grads.append(p.grad.abs().min())
    
    return layers, min_grads, ave_grads, max_grads