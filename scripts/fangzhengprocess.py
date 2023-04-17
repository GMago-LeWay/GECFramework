import json

result_file = "/home/liwei/workspace/GECProject/results/results/softmaskedbert-fangzhengtest-infer-0.json"
src_file = "/home/liwei/workspace/datasets/FangZhengTest/test.json"

with open(result_file, 'r') as f:
    results = json.load(f)

with open(src_file, 'r') as f:
    samples = json.load(f)

assert len(results) == len(samples)

with open('fangzhengmetrics/test.txt', 'w') as f:
    for i in range(len(results)):
        predict = results[i]
        src = samples[i]['text']
        tgt = samples[i]['label']
        assert len(predict) == len(src) == len(tgt)
        f.write(f"{src}\t{tgt}\t{predict}\n")
