#!/bin/bash
if [ "$dlgit" ]; then
    return
fi

export dlgit="dlgit.sh"

. dllog.sh

function dlgit_down_git(){
    local git_name="$1"
    local folder="$2"

    local pybase_git_folder="$folder/$git_name"
    dlfile_check_is_have_dir $pybase_git_folder

    if [[ $? -eq 0 ]]; then
        git clone git@git.oschina.net:darren_liu/$git_name.git "$pybase_git_folder"
    else
        $DLLOG_INFO "$1 git had been clone"
    fi
}

function dlgit_clone_git(){
    local user_name="$1"
    local git_name="$2"

    local folder="/home/$user_name/$git_name"
    dlfile_try_create_dir "$folder"
    dlgit_down_git $git_name $folder
}

if [ -n "$BASH_SOURCE" -a "$BASH_SOURCE" != "$0" ]
then
    echo other use
else # Otherwise, run directly in the shell
    echo fun self
fi

