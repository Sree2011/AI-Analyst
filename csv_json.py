# convert csv to json

import csv
import json
import pandas as pd

file = pd.read_csv("train.csv",sep=";")
gg = file.to_xml(parser="etree")

# save the xml file
with open("train.xml", "w") as f:
    f.write(gg)

