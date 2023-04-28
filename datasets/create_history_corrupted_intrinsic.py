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
from .create_intrinsic_hard import *


class HistoryCorruptIntrinsic(Annotator):
    def __init__(self, freebase_file: str, kge_dir: str, ext_entities: Dict):
        super().__init__(freebase_file=freebase_file,
                         kge_dir=kge_dir,
                         ext_entities=ext_entities)
        self.corrupted_turns = None
        self.corruption_cls = IntrinsicHard(freebase_file=freebase_file,
                                            kge_dir=kge_dir,
                                            ext_entities=ext_entities)

    # def corrupt_text(self, text, paths, verbose=True):
    #     if len(paths) > 0:
    #         entities = []
    #         relationship_embs = []
    #
    #         for ent in paths:
    #             if ent[2].lower() in text.lower() and len(entities) == 0:
    #                 entities.append(ent[2])
    #                 relationship_embs.append(self.kge.rel_embds[self.kge.rel2id[ent[1]]])
    #             elif ent[0].lower() in text.lower() and len(entities) == 0:
    #                 entities.append(ent[0])
    #                 relationship_embs.append(self.kge.rel_embds[self.kge.rel2id[ent[1]]])
    #             else:
    #                 pass
    #
    #         for ent, rels in self.kg[entities[0]].items():
    #             if ent == entities[0]:
    #                 continue
    #             else:
    #                 entities.append(ent)
    #                 relationship_embs.append(self.kge.rel_embds[self.kge.rel2id[list(rels.keys())[0]]])
    #
    #         relationship_embs = np.array(relationship_embs)
    #
    #         scores = cosine_similarity([relationship_embs[0]], relationship_embs[1:])
    #
    #         index = np.argsort(np.max(scores, axis=0))[-2]
    #
    #         corrupt_entity = entities[index]
    #         if verbose:
    #             print(50 * "=")
    #             print(f"Original entity was :{entities[0]}")
    #             print(f"Corrupt entity found is :{corrupt_entity}")
    #             print(f"Original response was:{text}")
    #
    #         corrupt_response = text.lower().replace(entities[0].lower(), corrupt_entity.lower())
    #
    #         if verbose:
    #             print(f"Corrupted response is: {corrupt_response}")
    #             print(50 * "=")
    #
    #         return corrupt_response, corrupt_entity, entities[0]
    #     else:
    #         return text.lower(), None, None

    def corrupt_history(self, df, dialogue_id):
        history = df[df["dialogue_id"] == dialogue_id].to_dict(orient="records")
        print("======= Corrupting history - <start> =======")
        corrupted_turns = []
        for i, h in enumerate(history):
            try:
                if len(h["knowledge_base"]["paths"]) > 0:
                    dialogue = {
                        "knowledge_base": {
                            "paths": h["knowledge_base"]["paths"]
                        },
                        "response": h["history"][-1]
                    }
                    # corrupt_response, corrupt_entity, og_entity = self.corrupt_text(text=h["history"][-1], paths=h["knowledge_base"]["paths"])
                    corrupt_response, corrupt_entities, og_entities = self.corruption_cls.create_dataset(dialogue)
            except:
                corrupt_response, corrupt_entities, og_entities = h["history"][-1], [], []

            corrupted_turns.append({
                "corrupt_response": corrupt_response,
                "og_response": h["history"][-1],
                "corrupt_entity": corrupt_entities,
                "og_entity": og_entities
            })

        print("======= Corrupting history - <end> =======")

        return corrupted_turns

    def create_dataset(self, dialogue, df, verbose=True):
        len_history = len(dialogue["history"])
        keys = ["corrupt_response", "og_response"]
        try:
            corrupt_history = self.corrupt_history(df, dialogue["dialogue_id"])
            corrupt_history = [h[random.choice(keys)] for h in corrupt_history][:len_history]
            corrupt_response, corrupt_entities, corrupt_relation, og_entities = self.corruption_cls.create_dataset(dialogue)

            return corrupt_history, corrupt_response, corrupt_entities, og_entities
        except Exception as e:
            corrupt_response, corrupt_entities, corrupt_relation,  og_entities = self.corruption_cls.create_dataset(dialogue)
            return [], corrupt_response, corrupt_entities, og_entities
