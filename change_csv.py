
"""

module prepares the source csv file for the csv file of the required format

"""

import numpy as np
import pandas as pd
import cv2
import csv

from os import listdir
from os.path import isfile, join, splitext

LOG_FILENAME = "log.txt"

input_img = "C:/Users/inet/Downloads/train/train"
input_df = "C:/Users/inet/Downloads/train/train.csv"

df = pd.read_csv(input_df)

row_number, _ = df.shape
row_number = str(row_number)

dict_for_image = {}

for item in listdir(input_img):

    image = cv2.imread(join(input_img, item), cv2.IMREAD_COLOR)

    x_0, y_0, _ = image.shape

    d = dict.fromkeys([item], (x_0, y_0))

    dict_for_image.update(d)


def decodeString(BoxesString):
    """
    Small method to decode the BoxesString
    """
    if BoxesString == "no_box":
        return np.zeros((0, 4))
    else:
        try:
            boxes = np.array([np.array([int(i) for i in box.split(" ")])
                              for box in BoxesString.split(";")])
            return boxes
        except:
            print(BoxesString)
            print("Submission is not well formatted. empty boxes will be returned")
            return np.zeros((0, 4))


boxes = [decodeString(item) for item in df["BoxesString"]]
image_list = df["image_name"].values

dict_for_box = {}
item = 0
for box in boxes:
    image_name = image_list[item]
    for i in box:

        d = dict.fromkeys([tuple(i)], (image_name + '.png'))

        dict_for_box.update(d)
    item += 1

with open("train_image.csv", 'w', newline='') as f:
    wr = csv.writer(f, delimiter=',', quotechar=',',
                    quoting=csv.QUOTE_MINIMAL)
    wr.writerows([('filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax')])
    count = 0
    for j in dict_for_box.keys():

        x_min = j[0]
        y_min = j[1]
        x_max = j[2]
        y_max = j[3]

        x_0, y_0 = dict_for_image.get(dict_for_box.get(j))

        wr.writerows([(count, dict_for_box.get(j), int("%.0f" % x_0), int("%.0f" % y_0), 'wheat_head',
                       float("%.2f" % x_min), float("%.2f" % y_min), float("%.2f" % x_max), float("%.2f" % y_max))])
        count += 1

test_df = pd.read_csv("train_image.csv")
print(test_df)





