import json
import re

import numpy as np
import pandas as pd

from khaiii import KhaiiiApi

class WordProcessor:

    def __init__(self):
        pass

    def re_sub(self, series):
        series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
        series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
        series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
        series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
        return series


    def get_token(self, title, tokenizer):
        if len(title) == 0 or title == ' ':  # 제목이 공백인 경우 tokenizer에러 발생
            return []

        result = tokenizer.analyze(title)
        result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
        return result


    def get_all_tags(self, df):
        tag_list = df['tags'].values.tolist()
        return tag_list


    def get_all_ttls(self, df):
        ttl_list = df['plylst_title'].values.tolist()
        return ttl_list


    def make_title_tokens(self, df):
        tokenizer = KhaiiiApi()

        df['plylst_title'] = self.re_sub(df['plylst_title'])
        all_ttl = self.get_all_ttls(df)

        token_ttl = [self.get_token(x, tokenizer) for x in all_ttl]
        df['ttl_token'] = token_ttl

        using_pos = ['NNG', 'SL', 'NNP', 'MAG', 'SN', 'XR']
        df['ttl_token_freq'] = df['ttl_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))

        return df


    def make_tag_tokens(self, df):
        tokenizer = KhaiiiApi()

        all_tag = self.get_all_tags(df)

        token_tag = [[self.get_token(x, tokenizer) for x in lst] for lst in all_tag]

        df.tag_token = token_tag

        def to_one_list(x):
            try:
                return [e[0] for e in x]
            except:
                return []

        df.tag_token = df.tag_token.apply(to_one_list)

        using_pos = ['NNG', 'SL', 'NNP', 'MAG', 'SN', 'XR']
        df['tag_token_freq'] = df['tag_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))

        return df


    def tokenize(self, df):
        tmp = self.make_title_tokens(df)
        tmp2 = self.make_tag_tokens(tmp)
        return tmp2


    def save(self, df):
        tokenized = self.tokenize(df)
        tokenized.to_json('full_token.json', orient='records')