#!/bin/sh

if [ "$setting" ]; then
    return
fi

export dlbase="setting.sh"

proc_name="tfrun"
venv_name="$proc_name""_env"
