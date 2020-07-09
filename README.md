# MelonPlaylistContinuation

## Usage
### &#183; neighbor&#46;py
```python
from neighbor import Neighbor

### 1. load data
train = pd.read_json("res/train.json")
val = pd.read_json("res/val.json")
# test = pd.read_json("res/test.json")

### 2. modeling
### 2.1 hyperparameter: pow_alpha, pow_beta
pow_alpha = 0.5
pow_beta = 0.3

### 3. range setting - Neighbor.predict()
### 3.1 range(start, end); if end == None, then range(start, end of val)
### 3.2 auto_save: boolean; False(default)
### 3.3 return type of Neighbor.predict() : pandas.DataFrame
pred = Neighbor(pow_alpha=pow_alpha, pow_beta=pow_beta, train=train, val=val).predict(start=0, end=None, auto_save=False)

### 4. save data
path = "."
fname = f"neighbor_a{int(pow_alpha * 10)}b{int(pow_beta * 10)}"
pred.to_json(f'{path}/{fname}.json', orient='records')
```
### &#183; knn&#46;py
```python
from knn import KNN

### 1. load data
train = pd.read_json("res/train.json")
val = pd.read_json("res/val.json")
# test = pd.read_json("res/test.json")

### 2. modeling
### 2.1 hyperparameter: k, rho, alpha, beta
### 2.2 variables: sim_songs, sim_tags, sim_normalize
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
### 3.3 return type of Neighbor.predict() : pandas.DataFrame
pred = KNN(k=k, rho=rho, alpha=alpha, beta=beta, \
            sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
            train=train, val=val, verbose=True, version_check=True).predict(start=0, end=None, auto_save=False)
# print(pred)

### 4. save data
path = "."
fname = f"knn_k{k}rho{int(rho * 10)}a{int(alpha * 10)}b{int(beta * 10)}_{sim_songs}{sim_tags}{sim_normalize}"
pred.to_json(f'{path}/{fname}.json', orient='records')
```
