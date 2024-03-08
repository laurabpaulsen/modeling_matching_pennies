# get path of this bash script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


# create virtual environment
bash $DIR/setup_env.sh

# activate virtual environment
source $DIR/env/bin/activate

# run python scripts
python $DIR/src/simulate.py
python $DIR/src/recover.py
python $DIR/src/plot_recovery.py
python $DIR/src/plot_individual.py
python $DIR/src/traceplots.py
