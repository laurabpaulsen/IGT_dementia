import stan
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations


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
    data1["group"] = -0.5
    data2["group"] = 0.5
    data2["sub"] = data2["sub"] + data1["sub"].nunique()

    data = pd.concat([data1, data2])

    # make a design matrix that includes the intercept and the group variable (one line for each subject)
    intercept = np.ones(data["sub"].nunique())

    group = np.array(data[data["trial"] == 1]["group"])
    
    design_matrix = np.vstack((intercept, group)).T

    # reshape choice and outcome to N_subj x T
    choice = np.array(data["choice"]).reshape(data["sub"].nunique(), -1)
    outcome = np.array(data["outcome"]).reshape(data["sub"].nunique(), -1)
    sign_out = np.array(data["sign_out"]).reshape(data["sub"].nunique(), -1)
    data_dict = {
        "choice" : choice.astype(int),
        "outcome" : outcome,
        "sign_out" : sign_out,
        "N": data["sub"].nunique(),
        "Tsubj": data.groupby("sub").size().values,
        "T": 100,
        "N_beta": 2,
        "X" : design_matrix

    }

    # fit the model
    model = stan.build(model_spec, data = data_dict)
    fit = model.sample(
        num_chains = 4, 
        num_samples = 1000,
        num_warmup = 1000)

    # get the estimated parameters
    df = fit.to_frame()


    # save the data
    if savepath:
        df.to_csv(savepath, index = False)



if __name__ == "__main__":
    path = Path(__file__).parent

    inpath = path / "simulated" / "group_lvl"
    outpath = path / "fit"

    if not outpath.exists():
        outpath.mkdir()

    with open(path.parent / "hierachical_IGT_ORL.stan") as f:
        model_spec = f.read()

    # make a list of tuples with all combinations of groups (don't want to compare group 1 with group 1 or group 1 with group 2 and group 2 with group 1)
    n_groups = 20
    compare_groups = list(combinations(range(1, n_groups + 1), 2))
 
    for group1, group2 in compare_groups:
        filename1 = f"ORL_simulated_group_{group1}_20_sub.csv"
        data1 = pd.read_csv(inpath / filename1)
        
        filename2 = f"ORL_simulated_group_{group2}_20_sub.csv"
        data2 = pd.read_csv(inpath / filename2)

        recover_group_level(
            data1 = data1, 
            data2 = data2, 
            model_spec = model_spec,
            savepath = outpath / f"param_rec_{group1}_{group2}.csv"
        )