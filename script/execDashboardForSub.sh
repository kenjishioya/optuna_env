#!/bin/bash
if [ ! -f .env ]; then
        echo ".env does not exist."
        exit 0
fi

eval $(egrep -v ‘^#’ .env | xargs)
docker run -it --rm -p 8080:8080 optuna_env bash -c "optuna-dashboard --host 0.0.0.0 mysql://$MYSQL_DB_USER:$MYSQL_DB_PASS@$MAIN_IP/$MYSQL_DB_NAME"
