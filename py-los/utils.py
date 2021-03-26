import json

# convert phonemes to a shorter json format (remove beginning/ending tags)
def convertPhonemes(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    for i in range(len(data["words"])):
        if data["words"][i]["case"] == "success":
            for j in range(len(data["words"][i]["phones"])):
                data["words"][i]["phones"][j]["phone"] = data["words"][i]["phones"][j]["phone"].split("_")[0]
    with open(fname+"result", 'w') as file:
        json.dump(data, file, indent=2)


# recursive addition method
def addPhoneme(vocab, phonemes, word):
    if len(phonemes)==1:
        vocab[phonemes[0]] = {"res": word}
        return vocab
    if phonemes[0] not in vocab:
        vocab[phonemes[0]] = {phonemes[1]: {}}
    addPhoneme(vocab[phonemes[0]], phonemes[1:], word)
    return vocab

# convert vocabulary to a phoneme tree
def makeVocabularyTree(fname):
    f = open("vocabulary.txt")
    vocab = {}
    for line in f.readlines():
        elts = line.split(' ')[1:]
        word = elts[0]
        phonemes = elts[1:]
        for i in range(len(phonemes)):
            phonemes[i] = phonemes[i].split("_")[0]
        addPhoneme(vocab, phonemes, word)
    print(len(vocab.keys())) # only 38
    with open("vocabulary_result.txt", 'w') as file:
        json.dump(vocab, file, indent=2)



# convertPhonemes("gatsby2.json")
makeVocabularyTree("vocabulary.txt")