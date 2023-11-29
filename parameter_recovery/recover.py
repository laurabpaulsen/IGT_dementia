import stan
from pathlib import Path
import numpy as np
import pandas as pd


def recover_group_level(data1, data2, model_spec, savepath = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    n_groups : int
        Number of groups to simulate.
    model_spec : str
        Stan model specification.
    savepath : Path, optional
        Path to save the fitted parameters to. The default is None.
    """
    data1["group"] = 0
    data2["group"] = 1
    data2["sub"] = data2["sub"] + data1["sub"].nunique()

    data = pd.concat([data1, data2])

    intercept = np.ones(int(len(data)))
    groups = data["group"] - 0.5 
    design_matrix = np.vstack((intercept, groups)).T

    print(design_matrix)

    data_dict = {
        "choice" : np.array(data["choice"]).astype(int),
        "outcome" : np.array(data["outcome"]),
        "sign_out" : np.array(data["sign_out"]),
        "subject":  np.array(data["sub"]).astype(int),
        "trial": np.array(data["trial"]).astype(int),
        "N": int(len(data)), #total number of trials
        "Nsubj": int(data["sub"].nunique()),
        "C": 4, # number of decks,
        "N_beta": 2,
        "X" : design_matrix

    }

    # fit the model
    model = stan.build(model_spec, data = data_dict)
    fit = model.sample(num_chains = 4, num_samples = 1000)

    # get the estimated parameters
    df = fit.to_frame()


    # save the data
    if savepath:
        df.to_csv(savepath, index = False)



if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fit"

    if not outpath.exists():
        outpath.mkdir()

    with open(path.parent / "hierachical_IGT_ORL.stan") as f:
        model_spec = f.read()

    # load in the simulated data
    filename1 = "ORL_simulated_group_1_20_sub.csv"
    data1 = pd.read_csv(path / "simulated" / filename1)

    filename2 = "ORL_simulated_group_2_20_sub.csv"
    data2 = pd.read_csv(path / "simulated" / filename1)


    recover_group_level(
        data1 = data1,
        data2 = data2, 
        model_spec = model_spec,
        savepath = outpath / f"param_rec.csv"
    )