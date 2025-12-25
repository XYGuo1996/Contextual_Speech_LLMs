import json
import random

source_json = "manifests/TED/ted_test_2_gt.json"
target_json = "manifests/TED/ted_test_2_gt_attacks_from_ted.json"

datas = []
with open(source_json, "r") as f:
    for line in f:
        data = json.loads(line)
        datas.append(data)

with open(target_json, "w") as f:
    for data in datas:
        while True:
            selected_task_prompt = datas[random.randint(0, len(datas)-1)]["task_prompt"]
            count_newline = selected_task_prompt.count("\n")
            if count_newline >= 4:
                break
        data["task_prompt"] = selected_task_prompt
        f.write(json.dumps(data, ensure_ascii=False) + "\n")