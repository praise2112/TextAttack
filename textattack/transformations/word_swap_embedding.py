"""
Word Swap by Embedding
============================================

"""
from textattack.shared.word_embedding import WordEmbedding
from textattack.transformations.word_swap import WordSwap


class WordSwapEmbedding(WordSwap):
    """Transforms an input by replacing its words with synonyms in the word
    embedding space."""

    PATH = "word_embeddings"

    def __init__(
        self,
        max_candidates=15,
        embedding_type="paragramcf",
        embedding_source=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_candidates = max_candidates
        self.embedding_type = embedding_type
        self.embedding_source = embedding_source
        self.embedding = WordEmbedding(
            embedding_type=embedding_type, embedding_source=embedding_source
        )

    def _get_replacement_words(self, word):
        """Returns a list of possible 'candidate words' to replace a word in a
        sentence or phrase.

        Based on nearest neighbors selected word embeddings.
        """
        try:
            word_id = self.embedding.word2ind(word.lower())
            nnids = self.embedding.nn(word_id, self.max_candidates)
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = self.embedding.ind2word(nbr_id)
                candidate_words.append(recover_word_case(nbr_word, word))
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []

    def extra_repr_keys(self):
        return ["max_candidates", "embedding_type"]


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word
