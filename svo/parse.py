# %%

from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from dataclasses import dataclass
from typing import List, Tuple, Optional

# %%


@dataclass
class ObjectPhrase:
    target: Token
    phrase: List[Token]

    def __str__(self):
        return " ".join([str(t) for t in self.phrase])

    def __hash__(self) -> int:
        return hash(str(self.target.idx) + str(self))


@dataclass
class VerbPhrase:
    target: Token
    subject: Optional[Token]
    object_: Optional[Token]
    phrase: Optional[List[Token]]

    def __str__(self):
        return " ".join([str(t) for t in self.phrase])

    def __hash__(self) -> int:
        return hash(str(self.verb.idx) + str(self))


def get_adp_phrase(t: Token) -> List[Token]:
    if t.head.pos_ == "ADP":
        try:
            first_child = next(t.children)

            if first_child.pos_ == "ADP":
                return [t.head, t, first_child]
        except StopIteration:
            pass

    return None


def is_adp_phrase(t: Token) -> bool:
    return get_adp_phrase(t) is not None


def get_object_phrases(doc: Doc, skip_determiner=False) -> List[ObjectPhrase]:
    ops = []
    last = -1

    for t in doc:
        is_root = t.pos_ == "NOUN" and t.dep_ == "ROOT"

        if t.pos_ == "PRON":
            continue

        if is_adp_phrase(t):
            continue

        if is_root or t.dep_ in ["nsubj", "nsubjpass", "dobj", "iobj", "pobj", "attr"]:
            subtree = list(t.subtree)
            start_idx = 0

            if t.pos_ == "PRON" and "cl" in t.head.dep_:
                t = t.head.head

            for i, c in enumerate(subtree):
                # if we're skipping determiners, offset the subtree
                if skip_determiner and c.pos_ == "DET":
                    start_idx = i + 1

                # if a new clause begins, end the subtree
                if "cl" in c.dep_ or c.pos_ in ["PRON", "ADP"]:
                    # only slice subtree if the child occurs after the target token
                    if t.i < c.i:
                        subtree = subtree[:i]
                        break

            if subtree[0].i > last:
                ops.append(
                    ObjectPhrase(
                        target=t,
                        phrase=list(doc[subtree[start_idx].i : subtree[-1].i + 1]),
                    )
                )
                last = subtree[-1].i

    return ops


# %%


def get_subject(t: Token) -> Token:
    if t.dep_ in ["acl"]:
        return t.head
    elif t.dep_ in ["advcl", "conj", "xcomp"]:
        return get_subject(t.head)

    # pronoun substitution
    pron_sub = t.head if t.dep_ == "relcl" else None

    for c in t.children:
        if c.dep_ in ["nsubj", "nsubjpass"]:
            if c.pos_ == "PRON":
                c = pron_sub

            return c


def get_prep_object(t: Token) -> Tuple[List[Token], Token]:
    for c in t.children:
        # if the prep-child is an object, assign it and exit loop
        if c.dep_ == "pobj":
            # if the object is part of an adposition, follow it to
            # the actual object
            if adp_phrase := get_adp_phrase(c):
                _, obj_token = get_prep_object(adp_phrase[-1])
                return adp_phrase, obj_token

            return t, c


def get_verb_phrases(doc: Doc) -> List[VerbPhrase]:
    vps = []

    for t in doc:
        # skip any non-verbs
        if t.pos_ != "VERB":
            continue

        vp = VerbPhrase(t, get_subject(t), None, None)

        for c in t.children:
            # if a child is a direct or indirect object of the
            # verb, we just assume that it's the object of interaction
            if c.dep_ in ["dobj", "iobj"]:
                vp.object_ = c
                vp.phrase = [t]

            # if a child is a preposition, we follow the children of that
            # preposition to find a preposition object
            elif c.dep_ == "prep":
                phrase_stop_token, obj_token = get_prep_object(c)
                vp.object_ = obj_token
                phrase = []

                for ti in t.subtree:
                    # only add tokens to phrase if the target token has been added
                    if len(phrase) > 0 or ti == t:
                        phrase.append(ti)

                    # stop iteration when we hit the stop token
                    if ti == phrase_stop_token:
                        break

                vp.phrase = phrase

            # add verb phrase if both subject and object is set
            if vp.subject and vp.object_:
                vps.append(vp)
                vp = VerbPhrase(t, get_subject(t), None, None)

    return vps


# %%


def get_svo(doc: Doc) -> Tuple[List[ObjectPhrase], List[VerbPhrase]]:
    return get_object_phrases(doc), get_verb_phrases(doc)


# %%

# import spacy

# nlp = spacy.load("en_core_web_trf")


# # %%

# caption = "a blonde and short woman have a cute poodle in her lap"
# caption = "a black honda motorcycle parked in front of a garage"
# caption = "a trio of dogs sitting in their owner's lap in a red convertible"
# caption = "a large passenger airplane flying through the air."

# get_object_phrases(nlp(caption)), get_verb_phrases(nlp(caption))
