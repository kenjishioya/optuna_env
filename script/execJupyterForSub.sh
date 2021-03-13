#!/bin/bash
if [ ! -f .env ]; then
        echo ".env does not exist."
        exit 0
fi

docker run -it --rm --env-file .env -p 8888:8888 --volume "$PWD"/src:/src --workdir /src optuna_env bash -c "jupyter notebook --allow-root"
