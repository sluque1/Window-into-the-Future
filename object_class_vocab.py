"""
Maps a free-form object-movement track name (assoc_info.json's ``name``, e.g.
"chopping board4" or "left half of third orange") onto the same closed
(sem_id, name) vocabulary Boxer/LVIS already uses for env-token class ids
(see model.py's ``class_embedding``, data/boxer_class_vocab.json).

Why reuse Boxer's vocabulary instead of a new one: assoc_info's names are
open-vocabulary, singleton-heavy (~40% of unique names occur exactly once)
and instance-suffixed ("bowl2"), which makes a from-scratch embedding table
both OOV-prone under LOOCV and wasteful. Mapping onto the existing 1082-class
vocabulary lets trajectory tokens share `class_embedding` with env tokens, so
"what a sponge is" is one embedding learned from both modalities at once.

Matching strategy (deliberately conservative -- see MATCHING.md-style notes
inline): exact match after normalization, else "does every content word of
a vocabulary phrase appear in the track name" (longest phrase wins). No
character-level fuzzy matching (e.g. difflib) -- it produces false positives
like "blade" -> "ladle" or "track" -> "truck" that would inject wrong
semantic signal, which is worse than leaving a track unmatched.

Coverage (measured against assoc_info.json): ~69% of movement tracks resolve
to a real class; the rest fall back to OOV_CLASS_IDX. This is a heuristic,
not a taxonomy -- e.g. "milk frother base" resolves via the substring "milk",
which is approximate. A frozen text encoder (see trajectory_tokens.py's
original design note) would generalize better but is separate future work.
"""

import json
import os
import re

_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "data", "boxer_class_vocab.json")

# class_embedding index convention (shared with env's `env_grid`, see model.py):
#   0                    = empty / padding (no track in this slot)
#   1 .. NUM_LVIS_CLASSES = matched Boxer/LVIS sem_id, shifted by +1
#   OOV_CLASS_IDX        = a real track whose name didn't match the vocabulary
#                          (trajectory-only; env never emits this index)
NUM_LVIS_CLASSES = 1220   # max sem_id (1219) + 1
OOV_CLASS_IDX = NUM_LVIS_CLASSES + 1   # 1221
NUM_OBJECT_CLASSES = OOV_CLASS_IDX + 1  # 1222 total class_embedding rows

_STOPWORDS = frozenset({"of", "the", "a", "an"})


def _normalize(name):
    name = name.strip().lower().replace("_", " ")
    # LVIS uses "word (qualifier)" to disambiguate homonyms, e.g. "orange (fruit)" vs
    # "orange (color)" -- track names never spell out the qualifier, so drop it for matching.
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = re.sub(r"\d+$", "", name).strip()   # strip trailing instance digits: "bowl2" -> "bowl"
    return re.sub(r"\s+", " ", name)


def _singular(word):
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


class _Vocab:
    """Lazily-built, module-cached (name -> sem_id) index with word-subset matching."""

    def __init__(self):
        with open(_VOCAB_PATH) as f:
            sem_to_name = json.load(f)
        self.exact = {}       # normalized phrase -> sem_id
        phrase_words = {}     # normalized phrase -> frozenset(singular words)
        for sem_id_str, name in sem_to_name.items():
            phrase = _normalize(name)
            sem_id = int(sem_id_str)
            self.exact[phrase] = sem_id
            phrase_words[phrase] = frozenset(_singular(w) for w in phrase.split())
        # Longest (most specific) phrase first, so e.g. "chopping_board" wins over "board".
        self.phrases_longest_first = sorted(phrase_words, key=lambda p: -len(phrase_words[p]))
        self.phrase_words = phrase_words
        self.sem_id_of = self.exact

    def match(self, raw_name):
        """Return a Boxer sem_id for `raw_name`, or None if nothing matches."""
        norm = _normalize(raw_name)
        if norm in self.exact:
            return self.exact[norm]
        words = {_singular(w) for w in norm.split() if w not in _STOPWORDS}
        if not words:
            return None
        for phrase in self.phrases_longest_first:
            if self.phrase_words[phrase] <= words:   # phrase's words are a subset of the name's
                return self.exact[phrase]
        return None


_vocab = None


def _get_vocab():
    global _vocab
    if _vocab is None:
        _vocab = _Vocab()
    return _vocab


def class_idx_for_name(raw_name):
    """class_embedding row for a free-form object name (see module docstring for the index convention)."""
    sem_id = _get_vocab().match(raw_name)
    return sem_id + 1 if sem_id is not None else OOV_CLASS_IDX
