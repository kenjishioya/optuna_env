# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
from optuna.storages import RDBStorage


def get_storage():
    main_ip = os.environ.get("MAIN_IP")
    if main_ip is None or main_ip == '0.0.0.0':
        main_ip = 'optuna_mysql'
    url = f'mysql://{os.environ.get("MYSQL_DB_USER")}:{os.environ.get("MYSQL_DB_PASS")}@{main_ip}:{os.environ.get("MYSQL_DB_PORT")}/{os.environ.get("MYSQL_DB_NAME")}'
    return RDBStorage(url=url)
