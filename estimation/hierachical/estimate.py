import stan
from pathlib import Path
import numpy as np
import pandas as pd


# local imports
import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.fit_model import fit_group_level_one_group

def load_behavioural_pooled(path):
    # load data Jacus
    AD_data1 = pd.read_csv(path / "data1" / "AD" / "data_AD_all_subjects.csv")
    HC_data1 = pd.read_csv(path / "data1" / "HC" / "data_HC_all_subjects.csv")
    
    # load data Chainay
    AD_data2 = pd.read_csv(path / "data2" / "alzheimer_disease.csv")
    HC_data2 = pd.read_csv(path / "data2" / "healthy_controls.csv")

    # merge data
    AD_data2["sub"] += AD_data1["sub"].max()
    HC_data2["sub"] += HC_data1["sub"].max()

    AD_data = pd.concat([AD_data1, AD_data2])
    HC_data = pd.concat([HC_data1, HC_data2])

    HC_data["sub"] += AD_data["sub"].max()

            
    AD_data["group"] = 1
    HC_data["group"] = 2

    return AD_data, HC_data



if __name__ in "__main__":
    path = Path(__file__).parent
    fit_path = path / "fit"
    if not fit_path.exists():
        fit_path.mkdir()

    # load data
    AD_data, HC_data = load_behavioural_pooled(path.parent)

    with open(path.parents[1] / "models" / "hierachical_IGT_ORL.stan") as f:
        model_spec = f.read()

    for data, group in zip([AD_data, HC_data], ["AD", "HC"]):
        outpath = fit_path / f"param_est_{group}_pooled.csv"
        summary_path = fit_path / f"param_est_{group}_summary_pooled.csv"

    
        summary = fit_group_level_one_group(
            data = data,
            model_spec = model_spec,
            savepath = outpath,
            summary = True
        )

        # save summary
        summary.to_csv(summary_path)
