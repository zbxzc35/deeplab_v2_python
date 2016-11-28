export LD_LIBRARY_PATH_TEMP=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export PROMPT_TEMP=$PROMPT
export PROMPT="%{$reset_color%}(deeplab_v2) "$PROMPT
