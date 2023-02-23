# %%

import numpy as np
import networkx as nx
from typing import List
from spacy.tokens.doc import Doc
from svo.parse import ObjectPhrase, VerbPhrase

# %%


def get_adjacency_matrix(
    objects: List[ObjectPhrase], verbs: List[VerbPhrase]
) -> np.array:
    n = len(objects)
    adj_matrix = np.zeros((n, n))
    obj_idx_map = {op.target: i for i, op in enumerate(objects)}

    for verb in verbs:
        adj_matrix[obj_idx_map[verb.subject], obj_idx_map[verb.object_]] = 1

    return adj_matrix


def get_networkx_graph(
    objects: List[ObjectPhrase], verbs: List[VerbPhrase]
) -> nx.DiGraph:
    g = nx.DiGraph()

    for o in objects:
        g.add_node(o)

    target_node = {n.target.idx: n for n in g.nodes}

    for v in verbs:
        g.add_edge(
            target_node[v.subject.idx],
            target_node[v.object_.idx],
            verb=" ".join([str(p) for p in v.phrase]),
        )

    return g


def draw_svo_networkx_graph(g: nx.DiGraph, node_color="lightblue"):
    pos = nx.layout.shell_layout(g)

    nx.draw(
        g,
        pos,
        with_labels=True,
        node_color=node_color,
        node_size=500,
        arrowsize=18,
        font_size=12,
    )
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels={(u, v): attr["verb"] for u, v, attr in g.edges(data=True)},
        font_size=12,
    )
