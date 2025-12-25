import json
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
import glob
import os

english_normalizer = EnglishTextNormalizer()

input_dir = "SpeechLLM/manifests/TED"

json_files = glob.glob(os.path.join(input_dir, '*.json'))

for json_file in json_files:
    datas = []
    with open(json_file, "r") as read:
        for line in read:
            data = json.loads(line)
            datas.append(data)
    
    with open(json_file, "w") as write:
        for data in datas:
            data["text"] = english_normalizer(data["text"])
            write.write(json.dumps(data, ensure_ascii=False) + "\n")