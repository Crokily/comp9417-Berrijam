import csv
import os
import random
from random import sample
import shutil
f = open('Labels-NeedsRespray-2024-03-25.csv')
f_csv = list(csv.reader(f))
header = f_csv[0]
for row in f_csv[1:]:
    print(row)
#print(f_csv)
"""
if os.path.exists("samples"):
    print("samples folder exists")
    exit(0)
os.makedirs("samples") 
"""   
for i in range(10):
    os.makedirs(f"samples/sample_{i}")
    rd=random.randint(1,10)
    samples = sample(f_csv[1:],rd)
    #print(f"{i}th folder:")
    #print(samples)
    os.makedirs(f"samples/sample_{i}/Yes")
    os.makedirs(f"samples/sample_{i}/No")
    for item in samples:
        if item[1]=="Yes":
            shutil.copy(item[0], f"samples/sample_{i}/Yes")
        else:
            shutil.copy(item[0], f"samples/sample_{i}/No")