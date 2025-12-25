import json

source_json_files = [
    "data/tedlium/TEDLIUM_release3/legacy_data/dev.json",
    "data/tedlium/TEDLIUM_release3/legacy_data/test.json",
    "data/tedlium/TEDLIUM_release3/legacy_data/train.json"
]

target_json_files = [
    "manifests/TED/ted_dev.json",
    "manifests/TED/ted_test.json",
    "manifests/TED/ted_train.json"
]

for target_json, source_json in zip(target_json_files, source_json_files):
    with open(target_json, "w") as write:
        with open(source_json, "r") as read:
            for line in read:
                data = json.loads(line)
                data["text"] = data["text"].replace(" '", "'")
                data["task_prompt"] = "Transcribe speech to text."
                data["prompt_template"] = "USER: <S> <P> ASSISTANT: <T>"
                write.write(json.dumps(data, ensure_ascii=False) + "\n")