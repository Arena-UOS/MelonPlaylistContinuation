# MelonPlaylistContinuation

## Version Updates
* neighbor&#46;py : version Neighbor-2.0 &#8594; version Neighbor-3.0
* neighbor_knn&#46;py : version NeighborKNN-1.0

## Upcoming Updates
* fix docstrings
* use load_json & write_json instead of pandas.read_json & pandas.to_json

## Usage
### &#183; neighbor&#46;py and neighbor_knn&#46;py &#8594; main&#46;py
```python
import numpy as np
import pandas as pd
from neighbor import Neighbor
from neighbor_knn import NeighborKNN
from data_util import *
from arena_util import load_json, write_json

### 1. data & preprocessing
### 1.1 load data
song_meta = pd.read_json("res/song_meta.json")
train = pd.read_json("res/train.json")
val = pd.read_json("res/val.json")
# test = pd.read_json("res/test.json")

### 1.2 convert "tag" to "tag_id"
tag_to_id, id_to_tag = tag_id_meta(train, val)
new_train = convert_tag_to_id(train, tag_to_id)
new_val   = convert_tag_to_id(val  , tag_to_id)


### 2. modeling : Neighbor
### 2.1 hyperparameters: pow_alpha, pow_beta
pow_alpha = 0.7
pow_beta = 0.0

### 2.2 run Neighbor.predict() : returns pandas.DataFrame
pred = Neighbor(pow_alpha=pow_alpha, pow_beta=pow_beta, \
                train=new_train, val=new_val, song_meta=song_meta).predict(start=0, end=None, auto_save=True)

### 3. post-processing
### 3.1 convert "tag_id" to "tag"
pred = convert_id_to_tag(pred, id_to_tag)
# print(pred)

### ==============================(save data)==============================
version = Neighbor.__version__
version = version[version.find('-') + 1: version.find('.')]
path = "."
fname1 = f"neighbor{version}_a{int(pow_alpha * 10)}b{int(pow_beta * 10)}"
pred.to_json(f'{path}/{fname1}.json', orient='records')
### ======================================================================

### 4. modeling : NeighborKNN
### 4.1 hyperparameters: k, rho, weights
### 4.2 parameters: sim_songs, sim_tags, sim_normalize
k = 100
rho = 0.4
weight_val_songs  = 0.5
weight_pred_songs = 1 - weight_val_songs
weight_val_tags   = 0.5
weight_pred_tags  = 1 - weight_val_tags
sim_songs = "idf"
sim_tags = "cos"
sim_normalize = False

### 4.3 run NeighborKNN.predict() : returns pandas.DataFrame
pred = NeighborKNN(k=k, rho=rho, \
            weight_val_songs=weight_val_songs, weight_pred_songs=weight_pred_songs, \
            weight_val_tags=weight_val_tags, weight_pred_tags=weight_pred_tags, \
            sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
            train=train, val=val, song_meta=song_meta, pred=pred).predict(start=0, end=None, auto_save=True)
# print(pred)

### ==============================(save data)==============================
version = NeighborKNN.__version__
version = version[version.find('-') + 1: version.find('.')]
path = "."
fname2 = f"neighbor-knn{version}_k{k}rho{int(rho * 10)}s{int(weight_val_songs * 10)}t{int(weight_val_tags * 10)}_{sim_songs}{sim_tags}{sim_normalize}"
pred.to_json(f'{path}/{fname2}.json', orient='records')
### ======================================================================

### 5. save data
path = "."
pred.to_json(f'{path}/{fname1}-{fname2}.json', orient='records')
```

### &#183; knn&#46;py
```python
from knn import KNN

### 1. load data
train = pd.read_json("res/train.json")
val = pd.read_json("res/val.json")
# test = pd.read_json("res/test.json")

### 2. modeling
### 2.1 hyperparameters: k, rho, alpha, beta
### 2.2 parameters: sim_songs, sim_tags, sim_normalize
k = 100
rho = 0.4
alpha = 0.5
beta = 0.5
sim_songs = "idf"
sim_tags = "cos"
sim_normalize = False

### 3. range setting - KNN.predict()
### 3.1 range(start, end); if end == None, then range(start, end of val)
### 3.2 auto_save: boolean; False(default)
### 3.3 return type of KNN.predict() : pandas.DataFrame
pred = KNN(k=k, rho=rho, alpha=alpha, beta=beta, \
            sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
            train=train, val=val, verbose=True, version_check=True).predict(start=0, end=None, auto_save=False)
# print(pred)

### 4. save data
version = KNN.__version__
version = version[version.find('-') + 1: version.find('.')]
path = "."
fname = f"knn{version}_k{k}rho{int(rho * 10)}a{int(alpha * 10)}b{int(beta * 10)}_{sim_songs}{sim_tags}{sim_normalize}"
pred.to_json(f'{path}/{fname}.json', orient='records')
```