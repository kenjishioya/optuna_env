#!/bin/bash
if [ ! -f .env ]; then
        echo ".env does not exist."
        exit 0
fi

if [ "$1" == "" ] || [ $# -gt 1 ]; then
        echo "Give me file name you wanna exec."
        exit 0
fi

if [ ! -f ./src/$1 ]; then
        echo "$1 dose not exist."
        exit 0
fi

docker run -it --rm --network optuna-net --env-file .env --volume "$PWD"/src:/src --workdir /src optuna_env bash -c "python $1"
