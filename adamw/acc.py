import sys
import numpy as np

print 'Usage: python acc.py Filename'
filename = sys.argv[1];
config = ""
best_acc = 100.0
with open(filename) as f:
    for line in f:
        if "learning_rate" in line:
            print(line)
        #if "warmup" in line:
            #print(line)
        if "accuracy" in line:
            print(line)
            #mystr = line.split();
            #acclist = mystr[-1].split('%');
            #acc = float(acclist[0]);
            #if acc < best_acc:
                #best_acc = acc
                #print(config)
                #print("lowest error rate: " + str(best_acc))
