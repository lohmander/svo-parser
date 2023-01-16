import spacy
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from dataclasses import dataclass
from typing import List, Optional


# spacy model
nlp = spacy.load("en_core_web_trf")


@dataclass
class ObjectPhrase:
    target: Token
    phrase: List[Token]


@dataclass
class VerbPhrase:
    verb: Token
    subject: Optional[Token]
    object_: Optional[Token]
    phrase: Optional[List[Token]]


def get_object_phrases(doc: Doc) -> List[ObjectPhrase]:
    ops = []
    last = -1

    for t in doc:
        is_root = t.pos_ == "NOUN" and t.dep_ == "ROOT"

        if is_root or t.dep_ in ["nsubj", "nsubjpass", "dobj", "iobj", "pobj"]:
            subtree = list(t.subtree)

            if t.pos_ == "PRON" and "cl" in t.head.dep_:
                t = t.head.head

            for i, c in enumerate(subtree):
                # if a new clause begins, end the subtree
                if "cl" in c.dep_:
                    subtree = subtree[:i]

            if subtree[0].i > last:
                ops.append(ObjectPhrase(target=t, phrase=list(doc[subtree[0].i:subtree[-1].i+1])))
                last = subtree[-1].i

    return ops


def get_subject(t: Token) -> Token:
    return t


def get_verb_phrases(doc: Doc) -> List[VerbPhrase]:
    vps = []

    for t in doc:
        if t.pos_ != "VERB":
            continue
        
        vp = VerbPhrase(t, get_subject(t), None, None)

        for c in t.children:
            if c.dep_ in ["dobj", "iobj"]:
                vp.object_ = c
                vp.phrase = [t]
            elif c.dep_ == "prep":
                for ci in c.children:
                    if ci.dep_ == "pobj":
                        vp.object_ = ci
                        vp.phrase = [t, c]
                        break

            if vp.subject and vp.object_:
                vps.append(vp)

    return vps

