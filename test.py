import jsonlines
import json

list = []

c_count = 0
java_count = 0
problems = set()
keywords = {'first line', 'First line', 'First Line'}
sol_count = 0
case_count = 0

with jsonlines.open('data/seq_data_python3.jsonl') as f:
    for idx, line in enumerate(f): 
        if len(line['solutions']) > 512:
            sol_count += 1
            continue
        source = line
        for t in line['test_cases']:
            if len(t) > 256:
                case_count += 1
                # continue
            # list.append(source)
        for t in line['private_tests']:
            if len(t) > 128:
                case_count += 1
                continue
            list.append(source)
        if idx > 50000: break

print(sol_count)
print(case_count)

print(len(list))
