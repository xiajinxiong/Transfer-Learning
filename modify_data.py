import re

# pattern = re.compile(r"[01]( ).*?")

with open("/home/tanghaihong/workspace/original/SST-2/test.tsv", "r+", encoding='utf-8') as f:
    lines = f.readlines()
    f.seek(0)
    for line in lines:
        if line[0] == '0':
            line = line.replace('0 ','0\t', 1)
        else:
            line = line.replace('1 ','1\t', 1)
        f.write(line)