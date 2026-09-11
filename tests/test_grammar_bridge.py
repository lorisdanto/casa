"""The grammar-recognizer bridge, and MARS on a constrained target.

CASA's recognizers are sequential: they consume tokens in order and remember how many they have
seen. MARS revisits prefixes in whatever order the trie sends it, including backwards, so the
bridge has to replay from the start whenever the requested prefix diverges from the one the
recognizer stands on. That replay is the part worth testing, because getting it wrong produces a
sampler that is subtly wrong rather than one that crashes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from casa.envelope_runtime import _RecognizerMask

VOCAB = 5
failures = []


class FakeRecognizer:
    """Mimics LlguidanceTokenRecognizer: sequential, with a consumed-token counter.

    Accepts any sequence that alternates even and odd token ids starting with an even one, which
    is enough structure to tell a correct replay from a broken one.
    """

    vocab_size = VOCAB

    def __init__(self):
        self.current_index = 0
        self.path = []
        self.resets = 0
        self.advanced = 0

    def reset(self):
        self.resets += 1
        self.current_index = 0
        self.path = []

    def _ok(self, seq):
        return all(t % 2 == i % 2 for i, t in enumerate(seq))

    def try_advance_token_ids(self, token_ids):
        new = token_ids[self.current_index:].tolist()
        self.advanced += len(new)
        cand = self.path + new
        if not self._ok(cand):
            return False
        self.path = cand
        self.current_index += len(new)
        return True

    def filter_vocab(self):
        want = len(self.path) % 2
        return [t for t in range(VOCAB) if t % 2 == want]

    def apply_token_bitmask(self, logits, bitmask):
        allowed = set(bitmask)
        for t in range(logits.shape[-1]):
            if t not in allowed:
                logits[0, t] = float("-inf")

    def is_accepting(self):
        return True


def check(label, cond, detail=""):
    if not cond:
        failures.append(label)
    print(f"  {'ok    ' if cond else 'FAIL  '}{label}{('  ' + detail) if detail else ''}")


print("\nMask content")
rec = FakeRecognizer()
mask = _RecognizerMask(rec, "L")
m0 = mask([], VOCAB)
check("empty prefix allows even tokens only",
      [float(x) for x in m0] == [0.0, float("-inf"), 0.0, float("-inf"), 0.0])
m1 = mask([0], VOCAB)
check("after one token allows odd tokens only",
      [float(x) for x in m1] == [float("-inf"), 0.0, float("-inf"), 0.0, float("-inf")])

print("\nConstruction rewinds the recognizer")
rec = FakeRecognizer()
rec.try_advance_token_ids(torch.tensor([0, 1]))   # someone else left it mid-path
mask = _RecognizerMask(rec, "L")
check("a shared recognizer is reset on construction", rec.path == [] and rec.resets == 1,
      f"path={rec.path} resets={rec.resets}")

print("\nExtending the path must not reset")
rec = FakeRecognizer()
mask = _RecognizerMask(rec, "L")
base_resets = rec.resets
mask([], VOCAB)
mask([0], VOCAB)
mask([0, 1], VOCAB)
mask([0, 1, 2], VOCAB)
check("four nested prefixes cost no further reset", rec.resets == base_resets,
      f"resets={rec.resets - base_resets} beyond construction")
check("each token consumed once", rec.advanced == 3, f"advanced={rec.advanced}")

print("\nDiverging must reset and replay")
before = rec.resets
mask([0, 3], VOCAB)  # diverges at position 1
check("divergence triggers exactly one reset", rec.resets == before + 1,
      f"resets={rec.resets}")
check("recognizer now stands on the new path", rec.path == [0, 3], f"path={rec.path}")

print("\nReturning to a shorter prefix of the current path")
rec = FakeRecognizer()
mask = _RecognizerMask(rec, "L")
mask([0, 1, 2], VOCAB)
before = rec.resets
mask([0], VOCAB)
check("shortening resets, since the matcher cannot rewind", rec.resets == before + 1)
check("path is the shorter prefix", rec.path == [0], f"path={rec.path}")

print("\nWidth mismatch between recognizer and tokenizer")
rec = FakeRecognizer()
mask = _RecognizerMask(rec, "L")
wide = mask([], VOCAB + 3)
check("padded entries are forbidden", len(wide) == VOCAB + 3
      and all(float(x) == float("-inf") for x in wide[VOCAB:]))
narrow = mask([], VOCAB - 2)
check("truncated to the tokenizer width", len(narrow) == VOCAB - 2)

print("\nA prefix the grammar rejects is a bug, not a silent zero")
rec = FakeRecognizer()
mask = _RecognizerMask(rec, "L")
try:
    mask([1], VOCAB)  # odd first token, invalid
    check("invalid prefix raises", False, "no exception")
except ValueError as e:
    check("invalid prefix raises", "should be impossible" in str(e))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all grammar bridge checks passed")
