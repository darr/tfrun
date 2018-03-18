#!/bin/sh

# determine base directory; preserve where you're running from
realpath=$(readlink -f "$0")
export basedir=$(dirname "$realpath") #export basedir, so that module shell can use it. log.sh. e.g.
export filename=$(basename "$realpath") #export filename, so that module shell can use it. log.sh. e.g.

export PATH=$PATH:$basedir/dlbase
export PATH=$PATH:$basedir/dlproc
export PATH=$PATH:$basedir

#base sh file
. dlbase.sh
#function sh file
. setting.sh
. create_env.sh


#proc_name="tfrun"
#venv_name="$proc_name""_env"
basepath=$(cd `dirname $0`; pwd)
venv_full_path=$basepath"/../$venv_name"
proc_full_path=$basepath"/../$proc_name"
echo $venv_full_path



if [ ! -d "$venv_full_path" ]; then
    create_virtual_env_folder
else
    source $venv_full_path/bin/activate
    echo $venv_full_path folder exist
fi

#python ../tf1.py
#python tensorflow/examples/tutorials/mnist/mnist.py
#tensorboard --logdir=../minst_log
#python ../TEST_MNI/mnist_test.py
#python ../TEST_MNI/mnist_hyperparameter.py
#python ../TEST_MNI/mnist_chose_hyperparameter.py
#python ../TEST_MNI/mnist_softmax_xla.py

#python ../mnist.py --max_steps=400
python ../mnist.py
#tensorboard --logdir=./tmp/log
