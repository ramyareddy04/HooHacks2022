import os
import pandas as pd
import numpy as np
import csv

with open('train.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)

    thewriter.writerow(['filename', 'label'])
    directory = os.path.join('data/train/openLeftEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
            thewriter.writerow([str(file), str("0")])

    directory = os.path.join('data/train/closedLeftEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
           thewriter.writerow([str(file), str("1")])

    directory = os.path.join('data/train/openRightEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
           thewriter.writerow([str(file), str("2")])

    directory = os.path.join('data/train/closedRightEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
           thewriter.writerow([str(file), str("3")])

f.close()

with open('trainBinary.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)

    thewriter.writerow(['filename', 'label'])
    directory = os.path.join('data/train/openLeftEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
            thewriter.writerow([str(file), str("0")])

    directory = os.path.join('data/train/closedLeftEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
           thewriter.writerow([str(file), str("1")])

    directory = os.path.join('data/train/openRightEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
           thewriter.writerow([str(file), str("0")])

    directory = os.path.join('data/train/closedRightEyes')
    for root, dirs, files in os.walk(directory):
        for file in files:
           thewriter.writerow([str(file), str("1")])

f.close()

data = pd.read_csv('train.csv')
print(data.head())
