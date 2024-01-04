import pandas as pd
from pathlib import Path

def load_IGT_excel_data(path):
    """
    Loads the IGT data from the excel file and returns it as a pandas dataframe. Each sheet is one participant.
    """
    data = pd.read_excel(path, sheet_name = None)

    # for each participant, add the participant number as a column
    for i, key in enumerate(data.keys()):
        data[key]["sub"] = int(i+1)

        # only keep the last 100 trials
        data[key] = data[key].iloc[-100:]

        # remove if there are any missing values in perte
        data[key] = data[key][~data[key]["perte"].isnull()]

        # calculate outcome
        data[key]["outcome"] = (data[key]["gain"] + data[key]["perte"])/100
        data[key]["outcome"] = data[key]["outcome"].astype("float")

        data[key]["choice"] = data[key]["Running[Trial]"].replace({"ListA": 1, "ListB": 2, "ListC": 3, "ListD": 4})
        data[key]["choice"] = data[key]["choice"].astype("int")

        data[key]["trial"] = range(1, data[key].shape[0]+1)
        data[key]["trial"] = data[key]["trial"].astype("int")
        
        data[key]["sign_out"] = data[key]["outcome"].apply(lambda x: 1 if x > 0 else -1)
        data[key]["sign_out"] = data[key]["sign_out"].astype("int")


        # if the participant does not have 100 trials, add missing trials
        if data[key].shape[0] < 100:
            missing_trials = 100 - data[key].shape[0]
            data[key] = pd.concat([data[key], pd.DataFrame({"sub": [i+1]*missing_trials, "trial": [0] * missing_trials, "choice": [0]*missing_trials, "outcome": [0]*missing_trials, "sign_out": [0]*missing_trials})], ignore_index = True)
    
    data = pd.concat(data, ignore_index = True)

    # only keep relevant columns
    data = data[["sub", "trial", "choice", "outcome", "sign_out"]]
    
    return data



if __name__ in "__main__":
    path = Path(__file__).parent

    HC = load_IGT_excel_data(path / "data_replicate" / "IGT_E-Data_controles.xlsx")
    HC.to_csv(path / "data_replicate" / "healthy_controls.csv", index = False)

    AD = load_IGT_excel_data(path / "data_replicate" / "IGT_E-Data_patients.xlsx")
    AD.to_csv(path / "data_replicate" / "alzheimer_disease.csv", index = False)


    
    
