# Copyright 2026 Rabel Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bell's guess written the way the speech-to-text writes its final transcript.

Why: under LiveKit's default turn detection (its turn detector, with preemptive generation on), when the final transcript
arrives after the turn has ended LiveKit restarts the early reply unless the final's string is exactly the early start's
string (livekit-agents 1.8.13, voice/audio_recognition.py: `transcript_changed = self._audio_transcript !=
self._audio_preflight_transcript`). Bell's guesses come in lower case with no punctuation, while speech-to-texts
send cased, punctuated finals, so a right guess was never kept there. (With the speech-to-text's own end of turn, LiveKit
compares words, `_transcripts_equivalent`, and either form is kept.)

So the guess is written from the speech-to-text's own text of the words so far, exactly as it last showed them, and only
the words Bell adds are formed here: "i" and the common contractions restored, "AM"/"PM" after an hour, the first word
capitalised when it opens a sentence; the turn closed with "?" when its last sentence opens like a question and "." otherwise. A guess written this way is kept when the
final has that form; when the speech-to-text writes the end of the turn otherwise (a comma, a name not yet heard, a number), it is
not, and LiveKit starts on the final as it does without Bell.
"""

from __future__ import annotations

import re

# the words Bell's guesses carry without their apostrophe (LiveKit's comparison drops it), restored; a word that is also an
# English word without one ("ill", "were", "well", "its", "hell", "shell", "wed", "id") is left out unless measured
_CONTRACTIONS = {
    w.replace("'", ""): w
    for w in (
        "i'm i'd i've i'll don't doesn't didn't can't couldn't won't wouldn't shouldn't isn't aren't wasn't weren't "
        "haven't hasn't hadn't that's what's there's here's where's who's how's let's you're you've you'll you'd "
        "they're they've they'll we're we've it's he's she's"
    ).split()
}
_QUESTION_OPENERS = frozenset(
    "can could would will do does did is are am was were should shall may might have has what whats how where when "
    "why who which whose whom isnt arent dont doesnt didnt wont wouldnt couldnt cant".split()
)
_ENDS = (".", "?", "!")
_HOURS = frozenset(
    "one two three four five six seven eight nine ten eleven twelve thirty fifteen fortyfive".split()
)
_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def norm_word(token: str) -> str:
    """A shown token in the form Bell's guesses and LiveKit's comparison use: punctuation dropped, case-folded."""
    return _PUNCT.sub("", token).casefold()


def _form(word: str) -> str:
    if word == "i":
        return "I"
    if word in _CONTRACTIONS:
        f = _CONTRACTIONS[word]
        return "I" + f[1:] if f.startswith("i'") else f
    return word


def _cap(w: str) -> str:
    return w[:1].upper() + w[1:]


def _sentence_opener(toks: list[str]) -> str:
    start = 0
    for i, t in enumerate(toks[:-1]):
        if t.endswith(_ENDS):
            start = i + 1
    return norm_word(toks[start]) if toks[start:] else ""


def final_form(shown: str, added: tuple[str, ...]) -> str:
    """The guess in the final's form: `shown` (the speech-to-text's text of the words so far, as it showed them) and the
    words Bell adds after them. With no shown text, the added words are the whole turn."""
    shown = shown.strip()
    if not added:
        return shown
    words = [_form(w) for w in added]
    before = [norm_word(t) for t in shown.split()[-1:]] + list(added)
    for i, w in enumerate(added):  # "five pm" -> "five PM": the hour's mark, after a number
        prev = before[i]
        if w in ("am", "pm") and (prev in _HOURS or prev.isdigit()):
            words[i] = w.upper()
    if not shown or shown.endswith(_ENDS):
        words[0] = _cap(words[0])
    toks = (shown.split() if shown else []) + words
    return " ".join(toks) + ("?" if _sentence_opener(toks) in _QUESTION_OPENERS else ".")
