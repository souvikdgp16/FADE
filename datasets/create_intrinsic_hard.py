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
from rank_bm25 import BM25Okapi
from .utils import *


class IntrinsicHard(Annotator):
    def __init__(self, freebase_file: str, kge_dir: str, ext_entities: Dict):
        super().__init__(freebase_file=freebase_file,
                         kge_dir=kge_dir,
                         ext_entities=ext_entities)

    def create_dataset(self, dialogue, verbose=True):
        try:
            if len(dialogue["knowledge_base"]["paths"]) > 0:
                og_entities = []
                og_triples = []
                triples = []
                docs = []

                for ent in dialogue["knowledge_base"]["paths"]:
                    if ent[2].lower() in dialogue["response"].lower():
                        if ent[2] not in og_entities:
                            og_entities.append(ent[2])
                            og_triples.append(ent)
                    elif ent[0].lower() in dialogue["response"].lower():
                        if ent[0] not in og_entities:
                            og_entities.append(ent[0])
                            og_triples.append(ent)
                    else:
                        pass

                replace_list = []

                for og_entity, og_triple in zip(og_entities, og_triples):
                    for ent, rel in self.kg[og_entity].items():
                        if ent == og_entity:
                            continue
                        else:
                            if og_triple[0] != ent and og_triple[1] != rel:
                                already_assigned = False
                                for s in replace_list:
                                    if ent in s[0] or rel in s[1] or ent in s[1] or rel in s[0]:
                                        already_assigned = True
                                    elif og_entity == s[1]:
                                        already_assigned = True

                                if not already_assigned:
                                    triples.append([ent, rel, og_entity])

                    for t in triples:
                        docs.append(f"{t[0]} {t[1]} {t[2]}")

                    tokenized_corpus = [doc.split(" ") for doc in docs]

                    bm25 = BM25Okapi(tokenized_corpus)
                    query = " ".join(og_triple)
                    tokenized_query = query.split(" ")
                    doc_scores = bm25.get_scores(tokenized_query)
                    index = np.argsort(doc_scores)[0]
                    corrupt_entity = triples[index][0]
                    corrupt_entity_rel = list(dict(triples[index][1]).keys())[0]

                    replace_list.append((og_entity, corrupt_entity, corrupt_entity_rel))

                corrupt_response = dialogue["response"].lower()

                print(replace_list)

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

                return corrupt_response, list(set([s[1] for s in replace_list])), list(set([s[2] for s in replace_list])), og_entities
            else:
                return dialogue["response"].lower(), None, None, None
        except Exception as e:
            return dialogue["response"].lower(), None, None, None
