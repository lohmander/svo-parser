import unittest
import spacy
from typing import Callable, List, Tuple

from svo import get_object_phrases, get_verb_phrases


nlp = spacy.load("en_core_web_trf")


class ComparisonTestCase(unittest.TestCase):
    def compare_all(
        self, method: Callable, getter: Callable, xy: List[Tuple[str, List[str]]]
    ):
        for x, y in xy:
            doc = nlp(x)
            ops = method(doc)
            self.assertListEqual([str(getter(op)) for op in ops], y)


class TestObjectPhrases(ComparisonTestCase):
    def test_object_target(self):
        self.compare_all(
            get_object_phrases,
            lambda op: op.target,
            [
                ("a man overlooking a crowd of inflatable boats", ["man", "crowd"]),
                ("a boy throws a ball to a dog", ["boy", "ball", "dog"]),
                (
                    "a chubby chef fries an egg while talking to a coworker",
                    ["chef", "egg", "coworker"],
                ),
                (
                    "there is a boy who throws some balls to a golden dog running on a grassy field",
                    ["boy", "balls", "dog", "field"],
                ),
                (
                    "a woman is running across the field towards the peach tree behind the mountain",
                    ["woman", "field", "tree"],
                ),
                (
                    "two swedish men walks down the street while snacking on ginger bread and throwing a ball to their dog",
                    ["men", "street", "bread", "ball", "dog"],
                ),
            ],
        )

    def test_object_phrase(self):
        self.compare_all(
            get_object_phrases,
            lambda op: " ".join([str(o) for o in op.phrase]),
            [
                (
                    "a tall and blonde boy throws a ball to a cute dog",
                    ["a tall and blonde boy", "a ball", "a cute dog"],
                ),
            ],
        )


class TestVerbPhrases(ComparisonTestCase):
    def test_verb_target(self):
        texts = [
            ("a man overlooking a crowd of inflatable boats", ["overlooking"]),
            ("a boy throws a ball to a dog", ["throws", "throws"]),
            (
                "a chubby chef fries an egg while talking to a coworker",
                ["fries", "talking"],
            ),
            (
                "there is a boy who throws some balls to a golden dog running on a grassy field",
                ["throws", "throws", "running"],
            ),
            (
                "a woman is running across the field towards the peach tree behind the mountain",
                ["running", "running"],
            ),
            (
                "two swedish men walks down the street while snacking on ginger bread and throwing a ball to their dog",
                ["walks", "snacking", "throwing"],
            ),
        ]

        for text, verbs in texts:
            doc = nlp(text)
            ops = get_verb_phrases(doc)

            self.assertListEqual([str(op.verb) for op in ops], verbs)

    def test_pronoun_substitution(self):
        self.compare_all(
            get_verb_phrases,
            lambda vp: vp.subject,
            [
                (
                    "the manager is talking on the phone while she eats an apple",
                    ["manager", "manager"],
                ),
                (
                    "the chef is chopping some onions while he is watching TV",
                    ["chef", "chef"],
                ),
                (
                    "the guard walks alongside the building with his coworker who chews gum",
                    [
                        "guard",
                        "guard",
                        "coworker",
                    ],  # walks alongside, walks with, chews "who" should be substituted with coworker
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
