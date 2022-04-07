# kaggle_practice
kaggleのコンペの問題を色々お手軽に試せるようなコードを作ってみる。kedroとか、pycaretとか、mlflow使ってローコードにやってみたい

# Usage
## 前提条件
- python: 3.8.13
    - pyenvによるバージョン管理推奨
- poetry: 1.1.13
- docker: 20.10.0

## 使い方
1. dockerのコンテナをビルドする（最初 or poetryのライブラリ更新時のみ）
```
docker-compose up -d --build
```
2. dockerのコンテナが起動しているか確認する（※起動していなかったら、`docker-compose up -d`）

```
docker-compose ps
```
3. dockerのコンテナに入る（出たい時はexitコマンドを打つ）
```
docker-compose exec analysis bash
```
4. （jupyterを使う場合）`./juoyter.sh`を実行する（※結局↓をやっているだけ）
```
jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
```

## ライブラリをインストールしたい場合
poetryで管理しているので、poetryに準拠してライブラリを追加
```
# 最初に必要なライブラリを全てインストールする場合
poetry install

# ライブラリ追加
poetry add [ライブラリ名]

# ライブラリ更新
poetry update [ライブラリ名]
```