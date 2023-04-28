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
from rank_bm25 import BM25Okapi
import heapq


class ExtrinsicSoft(Annotator):
    def __init__(self, freebase_file: str, kge_dir: str, ext_entities: Dict):
        super().__init__(freebase_file=freebase_file,
                         kge_dir=kge_dir,
                         ext_entities=ext_entities)
        self.index = None
        self.build_index()

    def build_index(self):
        index = {}
        for type in self.ext_entities:
            print(f"======= Building index for {type} ========")
            entities = self.ext_entities[type]
            docs = []

            for entity in entities:
                try:
                    kg_triples = self.kg[entity]
                    doc = []
                    for t in kg_triples:
                        doc.append(f"{t[0]} {t[1]} {t[2]}")

                    docs.append(" ".join(doc))
                except:
                    # print("Entity not found in KG!")
                    pass

            tokenized_corpus = [doc.split(" ") for doc in docs]
            # print(tokenized_corpus)
            try:
                ind = BM25Okapi(tokenized_corpus)
            except:
                ind = None

            index[type] = {
                "index": ind,
                "docs": docs,
                "entities": entities
            }

        self.index = index

    # def scrape_entities_from_text(self, text):
    #     entities_map = {}
    #
    #     text = self.nlp(text)
    #
    #     for word in text.ents:
    #         #             entities.append((word.text,word.label_))
    #         if word.label_ in list(entities_map.keys()):
    #             entities_map[word.label_].append(word.text)
    #         else:
    #             entities_map[word.label_] = [word.text]
    #
    #     return entities_map

    def get_type(self, text):
        text = self.nlp(text.strip())
        try:
            return text.ents[0].label_
        except:
            return None

    def create_dataset(self, dialogue, mode="random", verbose=True):
        try:
            if len(dialogue["knowledge_base"]["paths"]) > 0:
                # entities_map = self.scrape_entities_from_text(dialogue['response'])

                # init_types = list(entities_map.keys())

                og_entities = []
                og_triples = []

                for ent in dialogue["knowledge_base"]["paths"]:
                    if ent[2].lower() in dialogue["response"].lower():
                        if ent[2] not in og_entities:
                            type = self.get_type(ent[2])
                            if type:
                                og_entities.append((ent[2], type))
                                og_triples.append(ent)
                    elif ent[0].lower() in dialogue["response"].lower():
                        if ent[0] not in og_entities:
                            type = self.get_type(ent[0])
                            if type:
                                og_entities.append((ent[0], type))
                                og_triples.append(ent)
                    else:
                        pass

                replace_list = []

                for s, triple in zip(og_entities, og_triples):
                    try:
                        triples = self.kg[s[0]]
                        entity = s[0]
                        index = self.index[s[1]]
                        sum_scores = np.array([0] * len(self.index[s[1]]["docs"]))

                        for t in triples:
                            query = f"{t[0]} {t[1]} {t[2]}"
                            tokenized_query = query.split(" ")

                            doc_scores = index["index"].get_scores(tokenized_query)
                            sum_scores = sum_scores + doc_scores

                        avg_scores = sum_scores / len(triples)
                        max_score = max(avg_scores)
                        # query = f"{triple[0]} {triple[1]} {triple[2]}"
                        # tokenized_query = query.split(" ")
                        #
                        # doc_scores = index["index"].get_scores(tokenized_query)
                        #
                        # idx = list(doc_scores).index(max(doc_scores))
                        idx = list(avg_scores).index(max_score)
                        corrupt_entity = index["entities"][idx]

                        if corrupt_entity.lower() == entity.lower() or corrupt_entity.lower() in dialogue["response"].lower():
                            t = heapq.nlargest(2, avg_scores)
                            corrupt_entity = index["entities"][list(avg_scores).index(t[1])]

                        assigned = False

                        if len(replace_list) > 0:
                            for s in replace_list:
                                if corrupt_entity.lower() in s[0].lower() or corrupt_entity.lower() in s[1].lower() \
                                        or corrupt_entity.lower() in dialogue["response"].lower():
                                    assigned = True
                                    break
                                elif entity.lower() in s[0].lower() or entity.lower() in s[1].lower():
                                    assigned = True
                                    break
                                elif s[0].lower() in entity.lower() or s[1].lower() in entity.lower():
                                    assigned = True
                                    break
                                elif s[0].lower() in corrupt_entity.lower() or s[1].lower() in corrupt_entity.lower():
                                    assigned = True
                                    break

                        if not assigned:
                            replace_list.append((entity, corrupt_entity))
                    except Exception as e:
                        print(e)
                        pass

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
