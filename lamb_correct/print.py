import sys
import numpy as np

print 'Usage: python acc.py Filename'
filename = sys.argv[1];
config = ""
best_acc = 0.0
with open(filename) as f:
    for line in f:
        if "learning_rate " in line:
            config = line
        #if "warmup" in line:
            #print(line)
        if "accuracy" in line:
            line = line.replace('}', '')
            line = line.replace(',', '')
            mystr = line.split();
            #print(mystr)
            n1 = float(mystr[-1]);
            n2 = float(mystr[-3]);
            n3 = float(mystr[-5]);
            n = min(n1, n2, n3)
            if n > best_acc or n > 0.94:
                #config = line
                best_acc = n
                print(config)
                print(best_acc)
