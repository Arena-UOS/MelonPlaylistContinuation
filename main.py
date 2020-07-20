import numpy as np
import pandas as pd
from neighbor import Neighbor
from neighbor_knn import NeighborKNN
from data_util import *
from arena_util import load_json, write_json


import argparse

parser = argparse.ArgumentParser(description='User kNN')
parser.add_argument('--khaiii', '-k', type=bool, default=True, help='khaiii availability')
parser.add_argument('--train_path', '-t', type=str, default='res/train.json', help='path of train.json')
parser.add_argument('--val_path', '-v', type=str, default='res/val.json', help='path of val.json')
parser.add_argument('--song_path', '-s', type=str, default='res/song_meta.json', help='path of song_meta.json')
parser.add_argument('--val_conv_path', '-vcp', type=str, default='res/val_title.json', help='if khaiii doesnt work-windows')
parser.add_argument('--alpha', '-a', type=float, default=0.7, help='alpha')
parser.add_argument('--beta', '-b', type=float, default=0.0, help='beta')
parser.add_argument('--song_k', '-sk', type=int, default=100, help='how many song similarity to embrace')
parser.add_argument('--tag_k', '-tk', type=int, default=100, help='how many tag similairty to embrace')
parser.add_argument('--song_k_step', '-ss', type=int, default=10, help='how many steps to go further after song less than 100')
parser.add_argument('--tag_k_step', '-ts', type=int, default=10, help='how many steps to go further after tag less than 10')
parser.add_argument('--rho', '-r', type=float, default=0.4, help='rho')
parser.add_argument('--weight_val_songs', '-wvs', type=float, default=0.9, help='weights to validation song compared to train_song')
parser.add_argument('--weight_val_tags', '-wvt', type=float, default=0.7, help='weights to validation tag compared to train_tag')
parser.add_argument('--sim_songs', '-sims', type=str, default='idf', help='methods to calculate similarity of songs')
parser.add_argument('--sim_tags', '-simt', type=str, default='idf', help='methods to calculate similarity of tags')
parser.add_argument('--sim_normalize', '-simn', type=bool, default=False, help='to normalizae simlarity or not')
args = parser.parse_args()

### 1. data & preprocessing
### 1.1 load data
train_path = args.train_path
val_path   = args.val_path

song_meta = load_json(args.song_path)
train     = load_json(train_path)
val       = load_json(val_path)

song_meta = pd.DataFrame(song_meta)
train     = pd.DataFrame(train)
val       = pd.DataFrame(val)

if args.khaiii:
    from title_to_Tag import Title_to_tag
    ### 1.2 only_title chage to tags
    val = Title_to_tag(train1=train_path, val1=val_path).change()

else:
    val = load_json(args.val_conv_path)
### 1.3 convert "tag" to "tag_id"
tag_to_id, id_to_tag = tag_id_meta(train, val)
train = convert_tag_to_id(train, tag_to_id)
val   = convert_tag_to_id(val  , tag_to_id)


### 2. modeling : Neighbor
### 2.1 hyperparameters: pow_alpha, pow_beta
pow_alpha = args.alpha
pow_beta = args.beta

### 2.2 run Neighbor.predict() : returns pandas.DataFrame
pred = Neighbor(pow_alpha=pow_alpha, pow_beta=pow_beta, \
                train=train, val=val, song_meta=song_meta).predict(start=0, end=10, auto_save=True)
# print(pred)

### ==============================(save data)==============================
# version = Neighbor.__version__
# version = version[version.find('-') + 1: version.find('.')]
# path = "."
# fname1 = f"neighbor{version}_a{int(pow_alpha * 10)}b{int(pow_beta * 10)}"
# pred.to_json(f'{path}/{fname1}.json', orient='records')
# ### ======================================================================

### 3. modeling : NeighborKNN
### 3.1 hyperparameters: k, rho, weights
### 3.2 parameters: sim_songs, sim_tags, sim_normalize
song_k = args.song_k
tag_k  = args.tag_k
song_k_step = args.song_k_step
tag_k_step  = args.tag_k_step
rho = args.rho
weight_val_songs  = args.weight_val_songs
weight_pred_songs = 1 - weight_val_songs
weight_val_tags   = args.weight_val_tags
weight_pred_tags  = 1 - weight_val_tags
sim_songs = args.sim_songs
sim_tags  = args.sim_tags
sim_normalize = args.sim_normalize

### 3.3 run NeighborKNN.predict() : returns pandas.DataFrame
pred = NeighborKNN(song_k=song_k, tag_k=tag_k, rho=rho, \
                   song_k_step=song_k_step, tag_k_step=tag_k_step, \
                   weight_val_songs=weight_val_songs, weight_pred_songs=weight_pred_songs, \
                   weight_val_tags=weight_val_tags, weight_pred_tags=weight_pred_tags, \
                   sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
                   train=train, val=val, song_meta=song_meta, pred=pred).predict(start=0, end=10, auto_save=True)

### 4. post-processing
### 4.1 convert "tag_id" to "tag"
pred = convert_id_to_tag(pred, id_to_tag)
pred = to_list(pred)
write_json(pred, "results.json") # path???
# print(pred)

### ==============================(save data)==============================
# version = NeighborKNN.__version__
# version = version[version.find('-') + 1: version.find('.')]
# path = "."
# fname2 = f"neighbor-knn{version}_k{k}rho{int(rho * 10)}s{int(weight_val_songs * 10)}t{int(weight_val_tags * 10)}_{sim_songs}{sim_tags}{sim_normalize}"
# pred.to_json(f'{path}/{fname2}.json', orient='records')
### ======================================================================

