import unittest

from cli.lib.semantic_search import semantic_chunking


class TestSemanticChunking(unittest.TestCase):
    def test_empty_string(self):
        result = semantic_chunking("")
        self.assertEqual(result, [])

    def test_single_sentence(self):
        text = "Hello world."
        result: list[str] = semantic_chunking(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Hello world.")

    def test_multiple_sentences_default_chunk_size(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result: list[str] = semantic_chunking(text, chunk_size=1, overlap=0)
        self.assertEqual(len(result), 4)

        self.assertEqual(result[0], "First sentence.")
        self.assertEqual(result[1], "Second sentence.")
        self.assertEqual(result[2], "Third sentence.")
        self.assertEqual(result[3], "Fourth sentence.")

    def test_chunk_size_two_with_one_overlap(self):
        text = "First. Second. Third."
        result = semantic_chunking(text, chunk_size=2, overlap=1)
        self.assertEqual(len(result), 2)

        self.assertEqual(result[0], "First. Second.")
        self.assertEqual(result[1], "Second. Third.")

    def test_chunk_size_three_with_two_overlap(self):
        text = "A. B. C. D. E."
        result = semantic_chunking(text, chunk_size=3, overlap=2)
        self.assertEqual(len(result), 3)

        self.assertEqual(result[0], "A. B. C.")
        self.assertEqual(result[1], "B. C. D.")
        self.assertEqual(result[2], "C. D. E.")

    def test_various_sentence_delimiters(self):
        text = "First sentence! Second sentence? Third sentence."
        result = semantic_chunking(text, 1)
        self.assertEqual(len(result), 3)

    def test_chunk_size_larger_than_sentences(self):
        text = "Only one. Two sentences."
        result = semantic_chunking(text, chunk_size=10, overlap=0)
        self.assertEqual(len(result), 1)

    def test_no_overlap_vs_with_overlap(self):
        text = "A. B. C. D. E. F."
        no_overlap = semantic_chunking(text, chunk_size=2, overlap=0)
        with_overlap = semantic_chunking(text, chunk_size=2, overlap=1)
        self.assertGreaterEqual(len(with_overlap), len(no_overlap))

    def test_leading_trailing_whitespace(self):
        text = "   Leading and trailing whitespace.   "
        result = semantic_chunking(text, chunk_size=1, overlap=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Leading and trailing whitespace.")

    def test_no_punctuation(self):
        text = "This is a sentence without punctuation"
        result = semantic_chunking(text, chunk_size=1, overlap=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "This is a sentence without punctuation")

    def test_whitespace_only_string(self):
        text = "     "
        result = semantic_chunking(text, chunk_size=1, overlap=0)
        self.assertEqual(result, [])

    def test_empty_middle_sentence(self):
        text = "First sentence.    . Third sentence."
        result = semantic_chunking(text, chunk_size=1, overlap=0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "First sentence.")
        self.assertEqual(result[1], "Third sentence.")
