if [[ ! -f /.dockerenv ]]; then
	source ${HOME}/.anaconda3/etc/profile.d/conda.sh
	conda activate chainer5
fi

if [[ $GDB == "1" ]]; then
	PYTHON="gdb -ex run --args python"

elif [[ $PROFILE == "1" ]]; then
	PYTHON="python -m cProfile -o profile"

else
	PYTHON="python"

fi
SCRIPT="../main.py"

MODEL_TYPE=${MODEL_TYPE:-inception}

OPTS="${OPTS} --model_type $MODEL_TYPE"


export OMP_NUM_THREADS=2


DATA=${DATA:-/home/korsch/Data/info.yml}

if [[ -z $DATASET ]]; then
	echo "DATASET variable is missing!"
	exit -1
fi
