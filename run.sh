#!/bin/bash

. load_config.sh

# Define the path to the container and conda env
CONT="${ENV['cont']}"
PYENV="${ENV['python']}"

# Parse the incoming command
COMMAND="$@"

# Enter the container and run the command
SING="${ENV['exec']} exec --nv -i"
mounts=(${ENV[mounts]})
BS=""
for i in "${mounts[@]}";do
    if [[ $i ]]; then
       BS="${BS} -B $i:$i"
    fi
done

# add the repo path to "/project"
BS="${BS} -B ${PWD}:/project"

$SING $BS $CONT bash -c "source /project/$PYENV/bin/activate \
	&& cd $PWD \
        && exec $COMMAND \
        && source deactivate"
