"""
This script preprocesses the data for the parameter estimation. The data is stored in doc files for each participant.

The output of this script is a csv file for each group. The csv file contains the data of all participants of the group.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# import package for reading doc files
import textract

def doc2pandas(doc):
    """
    Converts the doc file to a pandas dataframe
    """
    # split doc file into lines
    lines = doc.split("\n")

    # find the index of the line starting with "Trial"
    trial_idx = [i for i, line in enumerate(lines) if line.startswith("Trial")][0]

    data = lines[trial_idx:]
    data = [line.split() for line in data]

    # create pandas dataframe using the first row as column names and the rest as data
    data = pd.DataFrame(data[1:], columns=data[0])

    # remove rows with NaN
    data = data.dropna()

    # convert data to numeric
    data = data.apply(pd.to_numeric, errors="ignore", downcast="float")
    
    # create outcome column - punishment minus the reward and scale by 100
    data["outcome"] = (data["Win"] + data["Lose"]) / 100

    # change column names
    data = data.rename(columns={"Deck": "choice", "Trial": "trial"})

    # change choice to numeric
    data["choice"] = data["choice"].replace({"A'": 1, "B'": 2, "C'": 3, "D'": 4})

    # sign of outcome column
    data["sign_out"] = np.sign(data["outcome"]).astype(int)

    data["trial"] = data["trial"].astype(int)

    # only keep relevant columns
    data = data[["choice", "outcome", "sign_out", "trial"]]

    return data

if __name__ == "__main__":
    path = Path(__file__).parent

    # data path
    data_path = path / "data"

    for group in ["AD", "HC", "MCI"]:
        group_path = data_path / group

        data = pd.DataFrame()
        
        for i, path in enumerate(group_path.iterdir()):
            if path.suffix == ".doc":
                # load doc 
                doc = textract.process(str(path)).decode("utf-8")

                # convert doc to pandas dataframe
                data_tmp = doc2pandas(doc)

                data_tmp["sub"] = i + 1

                data = pd.concat([data, data_tmp])

        # save data
        data.to_csv(group_path / f"data_{group}_all_subjects.csv", index=False)

    

