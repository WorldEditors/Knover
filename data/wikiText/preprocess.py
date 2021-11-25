import sys

new_line=[]
vocab_dict = dict()
#f_vocab = open("vocab.txt", "w")
is_empty=True
sys.stdout.write("src\ttgt\n")

for line in sys.stdin:
    tokens = line.strip().split(" ")
    eff_tokens = []
    for token in tokens:
        if(len(token) > 0):
            eff_tokens.append(token)
        if token not in vocab_dict:
            vocab_dict[token] = 0
    if(len(eff_tokens) < 1):
        is_empty=True
        continue
    if(len(eff_tokens) > 1 and eff_tokens[0] == "=" and eff_tokens[-1] == "=" and is_empty and len(new_line) > 1000) or len(new_line) > 1516:
        if(len(new_line) > 0):
            sys.stdout.write(" ".join(["\t"] + new_line))
            sys.stdout.write("\n")
            new_line = []
    else:
        if(len(new_line) > 0):
            new_line.append("[SEP]")
    new_line.extend(eff_tokens)
    is_empty=False

#for i, key in enumerate(vocab_dict):
#    f_vocab.write("%s\t%d\n"%(key, i))
#f_vocab.close()
