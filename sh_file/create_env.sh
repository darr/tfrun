#!/bin/sh

if [ "$create_env" ]; then
    return
fi

export dlbase="create_env.sh"

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


basepath=$(cd `dirname $0`; pwd)
venv_full_path=$basepath"/../$venv_name"
proc_full_path=$basepath"/../$proc_name"

echo $basepath
echo $venv_full_path
echo $proc_full_path
echo $proc_name


function create_virtual_env_folder(){
if [ ! -d "$venv_full_path" ]; then
    virtualenv $venv_full_path --no-site-packages
    source $venv_full_path/bin/activate
    pip install numpy --upgrade
    pip install maplotlib --upgrade
    pip install jupyter --upgrade
    pip install scikit-image --upgrde
    pip install librosa --upgrde
    pip install nltk --upgrade
    pip install keras --upgrade
    pip install --no-cache-dir tensorflow
else
    source $venv_full_path/bin/activate
    echo $venv_full_path folder exist
#    sudo pip install --index-url https://pypi.douban.com/simple ipython 
#    sudo pip install --index-url https://pypi.douban.com/simple --upgrade pip 
    #pip install --index-url https://pypi.douban.com/simple --upgrade pip

    pip --default-timeout=100 install numpy --upgrade
    pip --default-timeout=100 install maplotlib --upgrade
    #pip --default-timeout=100 install jupyter --upgrade
    pip --default-timeout=100 install scikit-image --upgrde
    pip --default-timeout=100 install librosa --upgrde
    pip --default-timeout=100 install nltk --upgrade
    pip --default-timeout=100 install keras --upgrade
    pip --default-timeout=100 install --no-cache-dir tensorflow
fi
}

create_virtual_env_folder

