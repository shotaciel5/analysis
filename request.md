problem.mdの問題をなるべくかしこく取り組みたいです。
まず最初に
train, test.csvともにデータの形式を変更してください。
現在
datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registerd, count
変更後
id, dteday, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, cnt
idは上から1,2,3,...
weekdayはカレンダーから計算
temp, atemp, humidityは各データ内で正規化

出力形式はsubmit.csvです。test.csvに対するcntで、id, cntの形式でお願いします。

実際にEDAをしてください。さらにそこから効果的な特徴量エンジニアリングやヒューリスティック的なことなんでも有効的なことは活用して予測モデルを作成してください。

再現性のためにEDAや分析モデル作成の手法や分析結果や予測結果などはマークダウンに随時記録してください。とにかく有効な可能性があることはなんでもしてください。

notebook形式ではなく、python形式で文性を進めてください。
