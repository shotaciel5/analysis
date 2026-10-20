# Bike Sharing Demand

- submit.csv: 提出用ファイル（id,cnt）
- REPORT.md: EDA、検証計画、モデル比較、最終評価、予測結果
- analysis.py: CSV変換から最終予測までの再現用Python
- verify_outputs.py: 形式変換、分割、予測、保存モデルの独立検証
- train.csv / test.csv: 指定の15列へ変換したデータ
- raw/: 変換前の原本。再実行でも上書きしません
- artifacts/: EDA集計・図、実験ログ、fold別予測、評価値、混合比、モデル、環境情報
- requirements.txt: 実行環境のライブラリバージョン

Python 3.12で実行します。

    python -m pip install -r requirements.txt
    python analysis.py
    python verify_outputs.py --models

既存の開発予測はキャッシュから読み込みます。再学習するには:

    python analysis.py --fresh

## 評価に関する前提

開発・最終ホールドアウトの各分割では、評価開始以降の実績を学習に使いません。
最終提出モデルは、コンペで配布されたtrain全体を既知の教師データとして再学習します。
したがって、各test時点までの情報だけで順次再学習するオンライン運用のシミュレーションではありません。
入力にないtest正解や外部の正解データは使いません。

毎月20日以降の正解はありません。9日間の時系列検証で本番の予測期間に近づけますが、
年末など未観測の日付や初期2011年の精度は独立には評価できません。
月内の日にちそのものはモデル特徴量から除外しています。

