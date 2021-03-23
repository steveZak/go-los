import json

# convert phonemes to a shorter json format (remove beginning/ending tags)
def convertPhonemes(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    for i in range(len(data["words"])):
        if data["words"][i]["case"] == "success":
            for j in range(len(data["words"][i]["phones"])):
                data["words"][i]["phones"][j]["phone"] = data["words"][i]["phones"][j]["phone"].split("_")[0]
    with open(fname+"final", 'w') as file:
        json.dump(data, file, indent=2)




# convertPhonemes("gatsby2res.json")