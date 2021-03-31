# go-los
The mute video speech detection project

The mute speech resconstruction is performed in 2 steps with a Encoder-Decoder type architecture.

Encoder

- Frame-by-frame phoneme classification using a CNN (complete).
- Utilising a CRNN architecture to achieve better accuracy (planned early April 2021)

Decoder

- Aligning the phoneme probabilities to words by incorporating semantic, POS and phoneme pattern matching (planned late April 2021).


Results demonstration

Building a web app for real-time mute video speech decoding (possibly late May).
- Model will be run in Go using a Tensorflow binding
- Aiming to achieve real-time performance

---

Addressed Encoder Challenges:
- Basic phoneme classification

Encoder Challenges planned to be addressed:
- The CRNN architecture is crucial to consider given some phonemes are diphthongs (such as 'aɪ' in price or 'aʊ' in mouth) and cannot be inferred from a single video frame
- Adaptation to different camera angles and mouth shapes and structure

Unaddressed Encoder Challenges:
- Adaptation to various accents and dialects (Requires compiling several pronunciations of the same word)

---

Decoder Challenges planned to be addressed:
- Homophone / similar phonetic pattern disambiguation with Conditional Random Fields
- Lexical disambiguation via context-dependent vector embeddings aids improving overall accuracy of next word prediction.
(e.g. ...hit the brakes / ...hit to break)

<!-- 1.) Record yourself saying a word and place it in the main directory.
2.) Run python3 pylos/train.py example.MOV -->