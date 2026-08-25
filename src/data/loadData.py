import pandas as pd
import os

def loadData(filePath):
    if not os.path.isfile(filePath):
        raise FileNotFoundError("File not found: " + filePath)

    return pd.read_csv(filePath)

