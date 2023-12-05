import numpy as np
import pandas as pd
import stan
import arviz as az

def fit_group_level(data, model_spec, savepath = None, summary = False):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data from both groups.
    model_spec : str
        Stan model specification.
    savepath : Path, optional
        Path to save the fitted parameters to. The default is None.
    """

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
        num_samples = 2000,
        num_warmup = 1000)

    # get the estimated parameters
    df = fit.to_frame()

    # save the data
    if savepath:
        df.to_csv(savepath, index = False)

    if summary:
        return az.summary(fit)



def fit_subject_level(data, model_spec, savepath = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    
    model_spec : str
        Stan model specification.
    savepath : Path, optional
        Path to save the fitted parameters to. The default is None.
    """

    data_dict = {
        "choice" : np.array(data["choice"]).astype(int),
        "outcome" : np.array(data["outcome"]),
        "sign_out" : np.array(data["sign_out"]),
        "T": int(len(data)), #total number of trials
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