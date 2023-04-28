import pandas as pd
from pprint import pprint
from tqdm import tqdm
import numpy as np
import ast
from itertools import chain
import pickle
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


class IntrinsicRepetitive(Annotator):
    def __init__(self, freebase_file: str, kge_dir: str, ext_entities: Dict):
        super().__init__(freebase_file=freebase_file,
                         kge_dir=kge_dir,
                         ext_entities=ext_entities)

    def history_entities(self, history, paths, response):
        repetitive_entities = []
        for h in history:
            for p in paths:
                if p[0].lower() not in response.lower():
                    if p[0] in h:
                        repetitive_entities.append(p[0])
                elif p[2].lower() not in response.lower():
                    if p[2] in h:
                        repetitive_entities.append(p[2])
                else:
                    pass

        return repetitive_entities

    def create_dataset(self, dialogue, verbose=True):
        try:
            if len(dialogue["knowledge_base"]["paths"]) > 0:
                repetitive_entities = self.history_entities(dialogue['history'], dialogue["knowledge_base"]['paths'], dialogue['response'])
                print(repetitive_entities)
                og_entities = []
                replace_list = []

                for ent in dialogue["knowledge_base"]["paths"]:
                    if ent[2].lower() in dialogue["response"].lower():
                        og_entities.append(ent[2])
                    elif ent[0].lower() in dialogue["response"].lower():
                        og_entities.append(ent[0])
                    else:
                        pass

                og_entities = list(set(og_entities))

                #corrupt_entity = repetitive_entities[0]
                corrupt_response = dialogue["response"].lower()

                for i, og_entity in enumerate(og_entities):
                    assigned = False
                    for s in replace_list:
                        if repetitive_entities[i] in s[1]:
                            assigned = True

                    if repetitive_entities[i] != og_entity and not assigned:
                        replace_list.append((og_entity, repetitive_entities[i]))

                for s in replace_list:
                    if verbose:
                        print(50 * "=")
                        print(f"Original entity was :{s[0]}")
                        print(f"Corrupt entity found is :{s[1]}")
                        print(f"Original response was:{corrupt_response}")

                    corrupt_response = corrupt_response.replace(s[0].lower(), s[1].lower())

                    if verbose:
                        print(f"Corrupted response is: {corrupt_response}")
                        print(50 * "=")

                return corrupt_response, list(set([s[1] for s in replace_list])), og_entities
            else:
                return dialogue["response"].lower(), None, None
        except Exception as e:
            # print(e)
            return dialogue["response"].lower(), None, None
