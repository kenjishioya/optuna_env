# optuna並列化環境の作成ツール

ハイパーパラメータ自動最適化フレームワークの[Optuna](https://optuna.org/)の並列化を簡単に実行するためのツールです。  
DockerでPython実行環境とMySQLを作成しハイパーパラメータのチューニングを行います。  
MySQLでチューニングの実行結果を保存し[並列化](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)を行います。  
  
![optuna_env説明図](https://user-images.githubusercontent.com/72649937/111029676-488ef080-8441-11eb-8312-46e3d132230e.png)  
  
上図の通り複数の端末を使用して並列でハイパーパラメータのチューニングが可能です。  
一端末内の別プロセスでの並列化も可能です。  
各端末で同じ環境をdockerで作成します。  
以降、MySQLを置く端末をメイン端末、メインに接続し並列で処理を行う端末をサブ端末と呼びます。  
  
optuna-dashbordによりリアルタイムで結果を確認できます。  

## 環境情報
Docker Engine v20.10.0  
Docker Compose v1.27.4  
M1 Macで動作確認  

## サンプル実行
以下の手順でサンプルを実行できます。

### メイン端末での実行
1. イメージ作成  
```
docker build . -t optuna_env
```
タグ名をoptuna_envにすること。

2. .envの作成  
.sample_envをコピーして.envファイルを作成してください。  
MYSQLの環境情報は任意で変更可能です。  
MAIN_IPはサブ端末で変更するので変更しないこと。  

3. DBの作成とJupyter、optuna-dashboardの起動  
```
docker-compose up
```
このコマンド実行でDBの作成とJupyter notebookとoptunaのダッシュボードを起動します。  
以下のURLからアクセス可能です。  
  
Jupyter notebook:  
http://127.0.0.1:8888/?token=<token\>  
※コンソールにこの体系でURLが表示されます。  
  
optunaのダッシュボード:  
http://127.0.0.1:8080/

4. 処理の実行  
サンプルの処理はJupyter notebookもしくはシェルスクリプトから実行できます。  
  
シェルスクリプトからの実行:  
```
sh script/execForMain.sh sample.py
```
※第一引数に実行したいファイル名を指定  

5. ダッシュボードで確認  
2で起動したダッシュボードでリアルタイムで結果を確認できます。

### サブ端末での実行
1. イメージ作成  
```
docker build . -t optuna_env
```
タグ名をoptuna_envにすること。

2. .envの作成  
.sample_envをコピーして.envファイルを作成してください。  
MYSQLの環境情報は任意で変更可能です。  
MAIN_IPは同一ネットワークのメイン端末のIPに変更します。  
例）MAIN_IP=192.168.0.0  

3. 処理の実行  
サンプルの処理はJupyter notebookもしくはシェルスクリプトから実行できます。  
  
Jupyter notebookの起動:  
```
sh script/execJupyterForSub.sh
```
http://127.0.0.1:8888/?token=<token\>  
※コンソールにこの体系でURLが表示されます。  
  
シェルスクリプトからの実行:  
```
sh script/execForSub.sh sample.py
```
※第一引数に実行したいファイル名を指定  
  
4. ダッシュボードで確認  
以下のスクリプトでダッシュボードを起動し、リアルタイムで結果を確認できます。  
  
optunaのダッシュボードの起動:
```
sh script/execDashboardForSub.sh
```
http://127.0.0.1:8080/  

## メモ
Jupyter notebook作成、編集すると.pyファイルが同時に作成、編集できます。  
.pyでの実行の方がおそらく早そうなので、Jupyter notebookでロジックを編集してスクリプトで実行が良さそう。
