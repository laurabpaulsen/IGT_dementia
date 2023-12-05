
# get location of this bash script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# activate environment one level up
source "${SCRIPT_DIR}/../env/bin/activate"

# run pipeline
python "${SCRIPT_DIR}/preprocess.py"
python "${SCRIPT_DIR}/estimate.py"
python "${SCRIPT_DIR}/plot.py"

# deactivate environment
deactivate