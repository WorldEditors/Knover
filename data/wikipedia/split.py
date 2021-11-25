import sys

split_number = 20
f_lst = []

start = 0
start_line = sys.stdin.readline()

for idx in range(split_number):
    f_lst.append(open("wiki_split.part-%02d"%idx, "w"))
    f_lst[-1].write(start_line)

for line in sys.stdin:
    idx = start % split_number
    f_lst[idx].write(line)
    start += 1

for idx in range(split_number):
    f_lst[idx].close()
