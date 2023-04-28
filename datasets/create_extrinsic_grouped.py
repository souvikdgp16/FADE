import pandas as pd
from pprint import pprint
from tqdm import tqdm
import numpy as np
import ast
from itertools import chain
import pickle
import random
import spacy
import networkx as nx
from typing import List, Tuple, Optional, Set, Dict
import os
from pathlib import Path
from collections import defaultdict
import pickle
from spacy.training import offsets_to_biluo_tags
from sklearn.metrics.pairwise import cosine_similarity
from .utils import *

GROUPS = [
    ["PERSON", "ORG", "NORP"],
    ["LOC", "GPE", "FAC"],
    ['PRODUCT', 'WORK_OF_ART', 'LAW'],
    ['QUANTITY', 'DATE', 'ORDINAL', 'CARDINAL', 'TIME', 'MONEY', 'PERCENT', 'EVENT'],
]


class ExtrinsicGrouped(Annotator):
    def __init__(self, freebase_file: str, kge_dir: str, ext_entities: Dict):
        super().__init__(freebase_file=freebase_file,
                         kge_dir=kge_dir,
                         ext_entities=ext_entities)

    def scrape_entities_from_text(self, text):
        entities_map = {}

        text = self.nlp(text)

        for word in text.ents:
            #             entities.append((word.text,word.label_))
            if word.label_ in list(entities_map.keys()):
                entities_map[word.label_].append(word.text)
            else:
                entities_map[word.label_] = [word.text]

        return entities_map

    def create_dataset(self, dialogue, verbose=True):
        try:
            if len(dialogue["knowledge_base"]["paths"]) > 0:
                entities_map = self.scrape_entities_from_text(dialogue['response'])

                init_types = list(entities_map.keys())
                replace_list = []

                #random_type = random.choice(list(set(list(entities_map.keys())) - set([init_type])))

                for typ in init_types:
                    og_entities = entities_map[typ]
                    for og_entity in og_entities:
                        for group in GROUPS:
                            if typ in group:
                                random_type = random.choice(list(set(group) - set([typ])))

                        corrupt_entity = random.choice(self.ext_entities[random_type])
                        if corrupt_entity == og_entity:
                            corrupt_entity = random.choice(self.ext_entities[random_type])

                        assigned = False

                        if len(replace_list) > 0:
                            for s in replace_list:
                                if corrupt_entity.lower() in s[0].lower() or corrupt_entity.lower() in s[1].lower() \
                                        or corrupt_entity.lower() in dialogue["response"].lower():
                                    assigned = True
                                    break
                                elif og_entity.lower() in s[0].lower() or og_entity.lower() in s[1].lower():
                                    assigned = True
                                    break
                                elif s[0].lower() in og_entity.lower() or s[1].lower() in og_entity.lower():
                                    assigned = True
                                    break
                                elif s[0].lower() in corrupt_entity.lower() or s[1].lower() in corrupt_entity.lower():
                                    assigned = True
                                    break

                        if not assigned:
                            replace_list.append((og_entity, corrupt_entity))

                corrupt_response = dialogue["response"].lower()

                print(replace_list)
                replace_list_final = []

                for s in replace_list:
                    if verbose:
                        print(50 * "=")
                        print(f"Original entity was :{s[0]}")
                        print(f"Corrupt entity found is :{s[1]}")
                        print(f"Original response was:{corrupt_response}")

                    old_corrupted_response = corrupt_response
                    corrupt_response = corrupt_response.replace(s[0].lower(), s[1].lower())

                    if verbose:
                        print(f"Corrupted response is: {corrupt_response}")
                        print(50 * "=")

                        if corrupt_response != old_corrupted_response:
                            replace_list_final.append(s)

                return corrupt_response, list(set([s[1] for s in replace_list_final])), list(set([s[0] for s in replace_list_final]))
            else:
                return dialogue["response"].lower(), None, None
        except Exception as e:
            # print(e)
            return dialogue["response"].lower(), None, None
