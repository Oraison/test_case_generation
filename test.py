import jsonlines

list = {}

count = 0

with jsonlines.open('data/seq_data2.jsonl') as f:
    for line in f:
        codeType = line['language']
        # print(type(codeType))
        
        if codeType in list :
            list[codeType] = list[codeType] + 1
        else:
            list[codeType] = 1

        count += 1

        if count % 10000 == 0:
            print(count / 10000)
        
print(list)