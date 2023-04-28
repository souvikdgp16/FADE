import numpy as np
import spacy
import os
import json
from itertools import chain
from collections import defaultdict
import networkx as nx
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Set, Dict


class Annotator:
    def __init__(self, freebase_file: str, kge_dir: str, ext_entities:Dict):
        self.nlp = spacy.load("en_core_web_sm")
        self.sent_transformer = SentenceTransformer('bert-base-nli-mean-tokens', device="cpu")
        self.kg = self.load_kg(freebase_file)
        self.kge = KnowledgeGraphEmbedding(kge_dir)
        self.ext_entities = ext_entities

    def extract_ent(self, text: str) -> List:
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def refine_node(self, node: str) -> str:
        if node == "Megamind":
            node = "MegaMind"

        return node

    def load_kg(self, freebase_file: str, verbose: bool = True) -> nx.Graph:
        G = nx.MultiGraph()
        incomplete_triples = 0
        total_triples = 0

        with open(freebase_file, "r") as f:
            for line in f.readlines():
                total_triples += 1

                if len(line.strip().split("\t")) < 3:
                    incomplete_triples += 1
                    continue
                head, edge, tail = line.strip().split("\t")
                head = self.refine_node(head)
                tail = self.refine_node(tail)
                if edge.startswith("~"):
                    edge = edge[1:]
                    src, dest = tail, head
                else:
                    src, dest = head, tail

                if not G.has_edge(src, dest, key=edge):
                    G.add_edge(src, dest, key=edge)

        if verbose:
            print("Number of incomplete triples {} out of {} total triples".format(incomplete_triples, total_triples))
            print("Number of nodes: {} | Number of edges: {}".format(G.number_of_nodes(), G.number_of_edges()))

        return G

    def generate_relation_embeddings(self, relation_path: str) -> Dict:
        with open(relation_path, "r") as f:
            relations = []
            for line in f.readlines():
                relations.append(line.strip())

        embs = self.sent_transformer.encode(relations)

        relation2emb = {}
        for i, emb in enumerate(embs):
            relation2emb[relations[i]] = emb

        return relation2emb

    def refine(self, edge):
        if edge.startswith("~"):
            edge = edge[1:]

        return edge


class KnowledgeGraphEmbedding:
    def __init__(self, kge_path: str, *dataset_paths, init_mean: float = 0.0, init_std: float = 1.0):
        self.kge_path = kge_path
        self.node2id, self.id2node = self._load_ids("entities.txt")
        self.rel2id, self.id2rel = self._load_ids("relations.txt")

        self.node_embds = self._load_and_stack("entity_embedding.npy")
        self.rel_embds = self._load_and_stack("relation_embedding.npy")

        if dataset_paths:
            new_ents = set()
            new_rels = set()
            for dataset_path in dataset_paths:
                if not dataset_path:
                    continue
                _ents, _rels = self._find_new_objs(dataset_path)
                new_ents.update(_ents)
                new_rels.update(_rels)

            if new_ents:
                self.node_embds = self._resize_embeddings(
                    new_ents, self.node2id, self.id2node, self.node_embds, init_mean, init_std
                )

            if new_rels:
                self.rel_embds = self._resize_embeddings(
                    new_rels, self.rel2id, self.id2rel, self.rel_embds, init_mean, init_std
                )

    def _load_and_stack(self, np_file: str):
        # breakpoint()
        embds = np.load(os.path.join(self.kge_path, np_file))
        return np.vstack((np.zeros(embds.shape[-1]), embds))

    def _load_ids(self, file_name: str) -> tuple:
        with open(os.path.join(self.kge_path, file_name)) as reader:
            ent2id = defaultdict(lambda: len(ent2id))
            ent2id[self.pad] = self.pad_id
            id2ent = {self.pad_id: self.pad}

            for line in reader:
                entity = line.strip()
                id = ent2id[entity]
                id2ent[id] = entity

        return dict(ent2id), id2ent

    @property
    def pad(self) -> str:
        return "<pad>"

    @property
    def pad_id(self):
        return 0

    def resize(
        self,
        new_ents: Optional[Set[str]] = None,
        new_rels: Optional[Set[str]] = None,
        dataset_path: str = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
    ) -> Tuple[int, int]:
        if dataset_path is None:
            assert new_ents is not None and new_rels is not None
        else:
            new_ents, new_rels = self._find_new_objs(dataset_path)

        if new_ents:
            self.node_embds = self._resize_embeddings(
                new_ents, self.node2id, self.id2node, self.node_embds, init_mean, init_std
            )

        if new_rels:
            self.rel_embds = self._resize_embeddings(
                new_rels, self.rel2id, self.id2rel, self.rel_embds, init_mean, init_std
            )

        return len(new_ents), len(new_rels)

    def _find_new_objs(self, dataset_path: str) -> tuple:
        new_ents = set()
        new_rels = set()
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                dialogue = json.loads(line.strip())
                if dialogue["knowledge_base"]:
                    for subj, pred, obj in dialogue["knowledge_base"]["paths"]:
                        if subj not in self.node2id:
                            new_ents.add(subj)
                        if obj not in self.node2id:
                            new_ents.add(obj)
                        if pred not in self.rel2id:
                            new_rels.add(pred)
        return new_ents, new_rels

    @classmethod
    def _resize_embeddings(
        cls,
        new_objs: Set[str],
        obj2id: Dict[str, int],
        id2obj: Dict[int, str],
        embds: np.ndarray,
        init_mean: float,
        init_std: float,
    ):
        new_embs = np.random.normal(init_mean, init_std, (len(new_objs), embds.shape[-1]))
        next_id = embds.shape[0]
        extended_embds = np.vstack((embds, new_embs))
        for obj in new_objs:
            obj2id[obj] = next_id
            id2obj[next_id] = obj
            next_id += 1

        return extended_embds

    def encode_node(self, node: str) -> int:
        return self.node2id[node]

    def decode_node(self, node_id: int) -> str:
        return self.id2node[node_id]

    def contains_node(self, node) -> bool:
        return node in self.node2id

    def encode_rel(self, relation: str) -> int:
        return self.rel2id[relation]

    def decode_rel(self, rel_id: int) -> str:
        return self.id2rel[rel_id]

    def contains_rel(self, relation: str) -> bool:
        return relation in self.rel2id