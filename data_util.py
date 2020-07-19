import numpy as np
import pandas as pd


def tag_id_meta(data):
    '''
    train, val : list of pandas.DataFrame
    @returns : (dictionary, dictionary)
    '''
    tag_to_id = {}
    id_to_tag = {}

    tag_id = 0
    for df in data:
        for idx in df.index:
            for tag in df["tags"][idx]:
                if not tag in tag_to_id:
                    tag_to_id[tag] = tag_id
                    id_to_tag[tag_id] = tag
                    tag_id += 1
    return tag_to_id, id_to_tag

def convert_tag_to_id(data, tag_to_id):
    '''
    data : pandas.DataFrame
    tag_to_id : dictionary
    '''
    data = data.copy()
    for idx in data.index:
        new_tags = []
        for tag in data["tags"][idx]:
            new_tags.append(tag_to_id[tag])
        data.at[idx, "tags"] = new_tags
    return data

def convert_id_to_tag(data, id_to_tag):
    '''
    data : pandas.DataFrame
    id_to_tag : dictionary
    '''
    data = data.copy()
    for idx in data.index:
        new_tags = []
        for tag_id in data["tags"][idx]:
            new_tags.append(id_to_tag[tag_id])
        data.at[idx, "tags"] = new_tags
    return data

def to_list(df):

    lst = []
    for idx in df.index:
        dct = {}
        dct["id"]    = df["id"][idx]
        dct["songs"] = df["songs"][idx]
        dct["tags"]  = df["tags"][idx]
        lst.append(dct)
    return lst



if __name__=="__main__":

    pass