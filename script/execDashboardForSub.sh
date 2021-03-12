#!/bin/bash
if [ ! -f .env ]; then
        echo ".env does not exist."
        exit 0
fi

docker run -it --rm --env-file .env optuna_env bash -c "optuna-dashboard optuna-dashboard --host 0.0.0.0 mysql://$MYSQL_DB_USER:$MYSQL_DB_PASS@$MAIN_IP/$MYSQL_DB_NAME"
