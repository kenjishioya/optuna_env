FROM python:3.8
USER root

RUN apt-get update
RUN apt-get install -y sudo
RUN apt-get install -y wget
RUN apt-get install -y libfreetype6-dev libatlas-base-dev liblapack-dev gfortran
RUN apt-get install -y cmake

RUN pip install -U pip
RUN pip install -U setuptools
RUN pip install numpy==1.19.5
RUN pip install scipy
RUN pip install sklearn
RUN pip install pandas
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install jupytext
RUN pip install xgboost
RUN pip install lightgbm
RUN pip install optuna
RUN pip install optuna-dashboard

# jupyter setting
RUN jupyter notebook --generate-config
RUN echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> /root/.jupyter/jupyter_notebook_config.py
RUN echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"' >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo 'c.NotebookApp.allow_remote_access = True' >> /root/.jupyter/jupyter_notebook_config.py

# mysqlclient
RUN apt-get install -y libssl-dev
RUN sudo apt-get install -y python3-dev libmariadb-dev
RUN pip install mysqlclient