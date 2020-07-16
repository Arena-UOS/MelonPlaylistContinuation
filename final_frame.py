from arena_util import *

def preprocess(train):
    pass

if __name__=="__main__":

    # 1. Data Loading
    DATA_PATH = '../res/'
    train = load_json(DATA_PATH + 'train.json')
    target = load_json(DATA_PATH + 'val.json')


    # 2. Preprocess
    # Don't touch target(rule)
    train_processed = preprocess(train)


    # 3. Predict
    results = []
    for line in target:
        
        # classify case
        if case == 1:
            pass
        else:
            pass


    # 4. Make .json
    write_json(result, 'results.json')

