# video-check-tool
動画のテロップチェックツール

## 概要
動画のテロップを解析し、NGワードや誤字を検出するツールです。

## 使用技術
- Python
- Streamlit

## 主な機能
- 動画アップロード（mp4 / mov）
- エラー表示
- チェック実行
- 結果表示
- ダウンロード機能

## 画面イメージ

## 起動方法
pip install -r requirements.txt
streamlit run app/app.py

## デモ
[NotionポートフォリオURL](https://www.notion.so/3435d4b2ca788032b0e0fc9f2f2b1ca3?source=copy_link)

##工夫した点
	•	非対応ファイルのエラーハンドリングを実装
	•	処理状況を可視化するUI設計（進捗表示）
	•	結果をダウンロードできる機能を実装

##今後の改善
	•	実際のテロップ解析との連携
	•	検出精度の向上

※ ルール定義ファイルには実運用データが含まれるため、公開用リポジトリではサンプルデータに置き換えています。
