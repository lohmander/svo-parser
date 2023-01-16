import unittest

from svo import nlp, get_object_phrases


class TestObjectPhrases(unittest.TestCase):
    def test_object_target(self):
        texts = [
            ("a man overlooking a crowd of inflatable boats", ["man", "crowd"]),
            ("a boy throws a ball to a dog", ["boy", "ball", "dog"]),
            ("a chubby chef fries an egg while talking to a coworker", ["chef", "egg", "coworker"]),
            ("there is a boy who throws some balls to a golden dog running on a grassy field", ["boy", "balls", "dog", "field"]),
            ("a woman is running across the field towards the peach tree behind the mountain", ["woman", "field", "tree"]),
            ("two swedish men walks down the street while snacking on ginger bread and throwing a ball to their dog", ["men", "street", "bread", "ball", "dog"]),
        ]

        for text, objs in texts:
            doc = nlp(text)
            ops = get_object_phrases(doc)

            self.assertListEqual([str(op.target) for op in ops], objs)

    def test_object_phrase(self):
        texts = [
            ("a tall and blonde boy throws a ball to a cute dog", ["a tall and blonde boy", "a ball", "a cute dog"]),
        ]

        for text, phrases in texts:
            doc = nlp(text)
            ops = get_object_phrases(doc)

            self.assertListEqual([" ".join([str(o) for o in op.phrase]) for op in ops], phrases)


if __name__ == "__main__":
    unittest.main()

