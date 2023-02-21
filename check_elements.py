import jsonlines
import json

list = []

count = 0

with jsonlines.open('data/seq_data2.jsonl') as f:
    for line in f:
        list.append(line)
        count += 1
        print(line)
        if(count >= 10):
            break

with open('data/seq_data_sample.jsonl', 'w',  encoding='utf-8') as write_file:
    for line in list : write_file.write(json.dumps(line, ensure_ascii='False') + '\n')
