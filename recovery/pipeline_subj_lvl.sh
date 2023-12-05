
# set the number of subjects to simulate
N_SUBJ=100

# get location of this bash script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# activate environment one level up
source "${SCRIPT_DIR}/../env/bin/activate"

# run pipeline
python "${SCRIPT_DIR}/simulate_subj_lvl.py" --n_subj ${N_SUBJ}
python "${SCRIPT_DIR}/recover_subj_lvl.py" --n_subj ${N_SUBJ}
python "${SCRIPT_DIR}/plot_subj_lvl.py" --n_subj ${N_SUBJ}

# deactivate environment
deactivate