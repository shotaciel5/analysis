"""Reproducible bike demand experiment. Run: python analysis.py"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
RAW = ROOT / "raw"
SEED = 20261020
COLS = ["id", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"]
FOLDS = ["2011-10", "2012-02", "2012-06", "2012-10"]
HOLDOUT = "2012-12"
CONFIGS = {
    "lgb_basic": dict(engine="lgb", features="basic", rounds=1000, leaves=31),
    "lgb_calendar": dict(engine="lgb", features="calendar", rounds=1000, leaves=31),
    "lgb_weather": dict(engine="lgb", features="full", rounds=1000, leaves=31),
    "lgb_smooth": dict(engine="lgb", features="full", rounds=1600, leaves=15),
    "cat_calendar": dict(engine="cat", features="calendar", rounds=1400, depth=6),
    "cat_weather": dict(engine="cat", features="full", rounds=1400, depth=6),
    "cat_deep": dict(engine="cat", features="full", rounds=1600, depth=8),
    "xgb_weather": dict(engine="xgb", features="full", rounds=1200, depth=4),
    "lgb_components": dict(engine="lgb", features="full", rounds=1200, leaves=15, components=True),
    "cat_components": dict(engine="cat", features="full", rounds=1400, depth=6, components=True),
}

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def table(frame):
    frame = frame.copy()
    for c in frame:
        if pd.api.types.is_float_dtype(frame[c]):
            frame[c] = frame[c].map(lambda v: f"{v:.5f}" if pd.notna(v) else "")
    return "| " + " | ".join(map(str, frame.columns)) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in frame.itertuples(index=False, name=None))

def score(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(p, 0)) - np.log1p(y)) ** 2)))

def prepare():
    OUT.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    norms = {}
    originals = {}
    for name in ["train", "test"]:
        source = ROOT / f"{name}.csv"
        backup = RAW / f"{name}.csv"
        if not backup.exists():
            check = pd.read_csv(source, nrows=1)
            if "datetime" not in check:
                raise ValueError("Original CSV unavailable; restore raw CSV before conversion.")
            shutil.copy2(source, backup)
        a = pd.read_csv(backup)
        originals[name] = a
        t = pd.to_datetime(a.datetime)
        z = pd.DataFrame({
            "id": np.arange(1, len(a) + 1), "dteday": t.dt.strftime("%Y-%m-%d"),
            "season": a.season, "yr": t.dt.year - 2011, "mnth": t.dt.month, "hr": t.dt.hour,
            "holiday": a.holiday, "weekday": (t.dt.dayofweek + 1) % 7,
            "workingday": a.workingday, "weathersit": a.weather,
            "temp": a.temp, "atemp": a.atemp, "hum": a.humidity, "windspeed": a.windspeed,
            "cnt": a["count"] if "count" in a else np.nan
        })
        norms[name] = {}
        for c in ["temp", "atemp", "hum"]:
            low, high = float(z[c].min()), float(z[c].max())
            norms[name][c] = {"min": low, "max": high}
            z[c] = (z[c] - low) / (high - low) if high != low else 0.0
        z[COLS].to_csv(source, index=False)
        assert z.columns.tolist() == COLS
        assert z.id.tolist() == list(range(1, len(z) + 1))
        assert z[["temp", "atemp", "hum"]].ge(0).all().all()
        assert z[["temp", "atemp", "hum"]].le(1).all().all()
        for c in ["temp", "atemp", "hum"]:
            scale = norms[name][c]
            np.testing.assert_allclose(z[c] * (scale["max"] - scale["min"]) + scale["min"],
                                       a["humidity" if c == "hum" else c], atol=1e-10)
    dump(OUT / "normalization.json", norms)
    dump(OUT / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "seed": SEED,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "lightgbm", "xgboost", "catboost", "joblib"]},
        "raw_sha256": {n: hashlib.sha256((RAW / f"{n}.csv").read_bytes()).hexdigest() for n in originals}
    })
    env = json.loads((OUT / "environment.json").read_text(encoding="utf-8"))
    (ROOT / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in env["packages"].items()) + "\n", encoding="utf-8")
    return originals["train"], originals["test"]

def eda(train, test):
    t = train.assign(date=pd.to_datetime(train.datetime))
    t["hr"] = t.date.dt.hour
    t["weekday"] = (t.date.dt.dayofweek + 1) % 7
    t["month"] = t.date.dt.strftime("%Y-%m")
    t["year"] = t.date.dt.year
    for name, keys in {"hour_workingday": ["hr", "workingday"], "monthly": ["month"],
                       "weather": ["weather"], "weekday": ["weekday"], "year": ["year"]}.items():
        t.groupby(keys).agg(n=("count", "size"), mean_count=("count", "mean"),
                           mean_casual=("casual", "mean"), mean_registered=("registered", "mean")).reset_index().to_csv(OUT / f"eda_{name}.csv", index=False)
    train.describe().to_csv(OUT / "eda_describe.csv")
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for work, part in t.groupby("workingday"):
        part.groupby("hr")["count"].mean().plot(ax=axs[0, 0], label=f"workingday={work}")
    axs[0, 0].set(title="Hourly demand by day type", ylabel="Mean count"); axs[0, 0].legend()
    t.groupby("month")["count"].mean().plot(ax=axs[0, 1], rot=65, title="Monthly hourly mean")
    axs[1, 0].hist(t["count"], bins=60)
    axs[1, 0].set(title="Count distribution", xlabel="Count")
    t.groupby("temp")["count"].mean().plot(ax=axs[1, 1], title="Temperature association (confounded)", ylabel="Mean count")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=150); plt.close(fig)
    combined = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    dates = pd.to_datetime(combined.datetime)
    expected = pd.date_range(dates.min(), dates.max(), freq="h")
    diagnostics = {
        "train_rows": len(train), "test_rows": len(test),
        "train_missing_cells": int(train.isna().sum().sum()), "test_missing_cells": int(test.isna().sum().sum()),
        "train_duplicate_timestamps": int(train.datetime.duplicated().sum()),
        "test_duplicate_timestamps": int(test.datetime.duplicated().sum()),
        "train_test_timestamp_overlap": len(set(train.datetime) & set(test.datetime)),
        "missing_hours_in_combined_calendar": len(expected.difference(pd.DatetimeIndex(dates))),
        "train_days": sorted(pd.to_datetime(train.datetime).dt.day.unique().tolist()),
        "test_days": sorted(pd.to_datetime(test.datetime).dt.day.unique().tolist()),
        "component_sum_equals_count": bool((train.casual + train.registered == train["count"]).all()),
        "zero_windspeed_rate": float((train.windspeed == 0).mean()),
        "weather_frequencies_train": {str(k): int(v) for k, v in train.weather.value_counts().items()},
        "weather_frequencies_test": {str(k): int(v) for k, v in test.weather.value_counts().items()},
        "target_quantiles": {str(k): float(v) for k, v in train["count"].quantile([0,.25,.5,.75,.95,.99,1]).items()}
    }
    dump(OUT / "eda_diagnostics.json", diagnostics)
    hour = pd.read_csv(OUT / "eda_hour_workingday.csv")
    year = pd.read_csv(OUT / "eda_year.csv")
    norm = json.loads((OUT / "normalization.json").read_text(encoding="utf-8"))
    report = f"""# 自転車需要予測の分析記録

## データ変換と保存
- raw/train.csv、raw/test.csvに入力原本を保存。train.csv、test.csvは指定の15列へ変換済み。
- idは各ファイルで1から連番。dtedayは日付、yrは2011=0/2012=1、weekdayは日曜=0〜土曜=6。
- temp、atemp、humはファイルごとに(min,max)でMin-Max正規化。係数はartifacts/normalization.json。
- testの未知cntは空欄。casual/registeredは指定列から除外し、原本だけに保持。
- モデルでは正規化係数から気象量を元の物理単位へ戻す。train/testで異なる正規化範囲を同じ尺度で扱い、検証時の正規化係数経由の将来情報も排除する。IDは予測特徴量に使わない。

## 実施したEDA
- train={len(train):,}行、test={len(test):,}行。欠損セルはともに0。
- 重複日時はtrain={diagnostics['train_duplicate_timestamps']}、test={diagnostics['test_duplicate_timestamps']}。両集合の日時重複={diagnostics['train_test_timestamp_overlap']}。
- 2011〜2012年の全時刻に対する未収録時間={diagnostics['missing_hours_in_combined_calendar']}。欠けた時間を需要ゼロと決めつけて補完しない。
- trainは毎月1〜19日、testは20日以降。ランダム分割では月内の先の情報を使えるため、時系列検証を採用。
- countの中央値={train['count'].median():.1f}、最大={train['count'].max()}。RMSLEに合わせlog1p目的変数を使う。
- casual + registered = countが全行で成立。利用者構成の異なる需要を別々に学習する候補も用意。
- 風速ゼロは{diagnostics['zero_windspeed_rate']:.2%}。ゼロを欠損と断定せず、ゼロ指標を特徴量候補にする。
- 気象カテゴリ4はtrainで{diagnostics['weather_frequencies_train'].get('4',0)}件。稀少カテゴリの評価は不安定。
- 温度と需要の関係は季節・時間帯との交絡を含む。単純相関を因果効果と解釈しない。

### 年別平均需要
{table(year)}

### 勤務日別・時間帯平均
{table(hour)}

![EDA](artifacts/eda.png)

### 正規化の係数
{table(pd.DataFrame([dict(split=s, column=c, **v) for s, cs in norm.items() for c,v in cs.items()]))}

## 特徴量と情報の時点
- basic: 月、年、曜日、時刻、季節、休日、勤務日、当該時刻に提供される気象。
- calendar: 経過日数、年内日、周期sin/cos、時刻×勤務日・曜日、通勤帯、月×勤務日、米国連邦祝日・前後日、年末日付を追加。
- full: 気温と体感温度の差、風速ゼロ、湿度ゼロ、気温×湿度、1/3/24時間前の気象・気象差を追加。
- 気象のラグは時刻キーで過去方向のみ参照。未来の実績需要、casual、registered、countを説明変数に入れない。
- testの既知気象も過去時刻のラグ計算に使用。未知の需要は一切参照しない。当該時刻の気象は問題で与えられた説明変数として扱う。
- 将来時刻の気象、全日集計、未来の需要を使うターゲットエンコーディングや需要ラグ、外部の正解データ取得は行わない。
- 連邦祝日はpandasのカレンダーから決定的に計算。元のholiday/workingdayも保持。

## 検証計画（モデル実行前に固定）
- 開発4fold: 2011-10、2012-02、2012-06、2012-10の11〜19日を評価。
- 各foldはその月10日23時以前のtrainの実績だけで学習。過去月でもtestに属する日の正解は存在せず使用しない。
- 学習対象日時は検証開始より厳密に前。9日間の連続予測で、実際の9〜12日間の予測に近づける。
- 最終ホールドアウト: 2012-12の11〜19日。モデルと混合比は開発4foldのみで選定し、この正解は選定に使わない。
- RMSLEを主指標とし、foldを結合した予測で算出。単純なfold平均も別記。
- 反復回数は事前固定。検証/ホールドアウトでのearly stoppingを使わない。
- 最終ホールドアウトも平常期の9日間であり、年末・悪天候・本番順位の精度を保証しない。
- 同一月末へ予測する実運用との差、少数foldによるばらつき、2011初期の短い履歴を制約として残す。

## 実行ログ
EDAと検証計画を保存済み。以降の実験はartifacts/experiment_log.jsonlへ逐次追記。
"""
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")

def features(train, test):
    converted = pd.concat([pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")], ignore_index=True)
    norms = json.loads((OUT / "normalization.json").read_text(encoding="utf-8"))
    for start, end, name in [(0,len(train),"train"), (len(train),len(converted),"test")]:
        for c in ["temp","atemp","hum"]:
            s = norms[name][c]
            converted.loc[start:end-1,c] = converted.loc[start:end-1,c] * (s["max"] - s["min"]) + s["min"]
    a = converted
    dt = pd.to_datetime(a.dteday) + pd.to_timedelta(a.hr,unit="h")
    x = a[["season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"]].copy()
    basic = x.columns.tolist()
    x["elapsed_day"] = (dt - pd.Timestamp("2011-01-01")).dt.total_seconds()/86400
    x["dayofyear"] = dt.dt.dayofyear
    x["weekend"] = dt.dt.dayofweek.ge(5).astype(int)
    x["hour_workday"] = a.hr + 24*a.workingday
    x["hour_weekday"] = a.hr + 24*a.weekday
    x["hour_season"] = a.hr + 24*(a.season-1)
    x["month_workday"] = a.mnth + 12*a.workingday
    x["hour_year"] = a.hr + 24*a.yr
    x["rush_morning"] = (a.hr.between(7,9) & a.workingday.eq(1)).astype(int)
    x["rush_evening"] = (a.hr.between(16,19) & a.workingday.eq(1)).astype(int)
    x["night"] = a.hr.between(0,5).astype(int)
    for col, period in [("hr",24),("weekday",7),("dayofyear",365.25),("mnth",12)]:
        x[f"{col}_sin"] = np.sin(2*np.pi*x[col]/period)
        x[f"{col}_cos"] = np.cos(2*np.pi*x[col]/period)
    holidays = USFederalHolidayCalendar().holidays("2010-12-01","2013-01-15")
    date = dt.dt.normalize()
    x["federal_holiday"] = date.isin(holidays).astype(int)
    x["pre_federal_holiday"] = (date + pd.Timedelta(days=1)).isin(holidays).astype(int)
    x["post_federal_holiday"] = (date - pd.Timedelta(days=1)).isin(holidays).astype(int)
    x["yearend"] = ((dt.dt.month.eq(12) & dt.dt.day.ge(24)) | (dt.dt.month.eq(1) & dt.dt.day.le(2))).astype(int)
    calendar = x.columns.tolist()
    x["feels_minus_temp"] = a.atemp - a.temp
    x["wind_zero"] = a.windspeed.eq(0).astype(int)
    x["humidity_zero"] = a.hum.eq(0).astype(int)
    x["temp_humidity"] = a.temp*a.hum/100
    weather = a[["temp","atemp","hum","windspeed","weathersit"]].copy()
    weather.index = pd.DatetimeIndex(dt)
    for lag in [1,3,24]:
        values = weather.reindex(pd.DatetimeIndex(dt - pd.Timedelta(hours=lag))).reset_index(drop=True)
        for c in ["temp","hum","windspeed","weathersit"]:
            x[f"{c}_lag{lag}"] = values[c]
        x[f"temp_change{lag}"] = a.temp - values.temp
    return x.astype(float), {"basic":basic,"calendar":calendar,"full":x.columns.tolist()}, pd.Series(dt)

def split(dt, month):
    start = pd.Timestamp(f"{month}-11")
    end = pd.Timestamp(f"{month}-20")
    it = np.flatnonzero(dt.lt(start))
    iv = np.flatnonzero(dt.ge(start) & dt.lt(end))
    assert dt.iloc[it].max() < dt.iloc[iv].min()
    return it, iv

def fit_predict(cfg, x, y, it, ip, category_names, seed=SEED):
    common = dict(random_state=seed, n_jobs=4)
    if cfg["engine"] == "lgb":
        model = LGBMRegressor(n_estimators=cfg["rounds"], learning_rate=.025,
            num_leaves=cfg["leaves"], min_child_samples=25, colsample_bytree=.9,
            subsample=.9, subsample_freq=1, reg_lambda=3, verbosity=-1,
            deterministic=True, force_col_wise=True, **common)
        model.fit(x.iloc[it], np.log1p(y[it]))
        p = model.predict(x.iloc[ip])
    elif cfg["engine"] == "xgb":
        model = XGBRegressor(n_estimators=cfg["rounds"], learning_rate=.025,
            max_depth=cfg["depth"], min_child_weight=8, subsample=.9,
            colsample_bytree=.9, reg_lambda=5, objective="reg:squarederror",
            tree_method="hist", **common)
        model.fit(x.iloc[it], np.log1p(y[it]))
        p = model.predict(x.iloc[ip])
    else:
        x = x.copy()
        cats = [c for c in category_names if c in x]
        for c in cats:
            x[c] = x[c].fillna(-1).astype(int)
        model = CatBoostRegressor(iterations=cfg["rounds"], learning_rate=.035,
            depth=cfg["depth"], loss_function="RMSE", l2_leaf_reg=5, random_seed=seed,
            thread_count=4, verbose=False, allow_writing_files=False, cat_features=cats)
        model.fit(x.iloc[it], np.log1p(y[it]))
        p = model.predict(x.iloc[ip])
    return np.maximum(np.expm1(np.clip(p, -10, 15)),0), model

CATEGORIES = ["season","yr","mnth","hr","holiday","weekday","workingday","weathersit",
              "weekend","hour_workday","hour_weekday","hour_season","month_workday","hour_year",
              "federal_holiday","pre_federal_holiday","post_federal_holiday"]

def candidate(name, x, groups, train, it, ip, save=False):
    cfg = CONFIGS[name]
    xx = x[groups[cfg["features"]]]
    targets = ["casual","registered"] if cfg.get("components") else ["count"]
    p = np.zeros(len(ip))
    imps = []
    for target in targets:
        pred, model = fit_predict(cfg,xx,train[target].to_numpy(),it,ip,CATEGORIES)
        p += pred
        imp = pd.DataFrame({"feature":xx.columns,"importance":model.feature_importances_,"target":target,"model":name})
        imps.append(imp)
        if save:
            joblib.dump(model, OUT / "models" / f"{name}_{target}.joblib")
    return p, pd.concat(imps)

def baseline(train,x,it,ip):
    z = x.iloc[it][["hr","workingday","mnth"]].copy()
    z["log_count"] = np.log1p(train["count"].to_numpy()[it])
    profile = z.groupby(["hr","workingday","mnth"]).log_count.mean()
    fallback = z.groupby(["hr","workingday"]).log_count.mean()
    query = x.iloc[ip]
    result = []
    for row in query[["hr","workingday","mnth"]].itertuples(index=False,name=None):
        result.append(profile.get(row, fallback.get(row[:2],z.log_count.mean())))
    return np.expm1(result)

def log_event(obj):
    with (OUT / "experiment_log.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False)+"\n")
    print(json.dumps(obj,ensure_ascii=False),flush=True)

def run_experiments(train,test):
    x, groups, dt = features(train,test)
    dtrain = dt.iloc[:len(train)]
    pred_dir = OUT / "validation_predictions"; pred_dir.mkdir(exist_ok=True)
    rows = []; pieces = []
    for month in FOLDS:
        it, iv = split(dtrain,month)
        pieces.append(iv)
        for name in ["baseline"] + list(CONFIGS):
            path = pred_dir / f"{month}_{name}.csv"
            signature = hashlib.sha256(json.dumps({"cfg":CONFIGS.get(name),"version":1,"groups":groups},sort_keys=True).encode()).hexdigest()
            if path.exists():
                old = pd.read_csv(path)
                assert old.signature.eq(signature).all()
                np.testing.assert_array_equal(old.row,iv)
                p = old.prediction.to_numpy()
            else:
                start = time.monotonic()
                if name == "baseline":
                    p = baseline(train,x,it,iv)
                else:
                    p,_ = candidate(name,x,groups,train,it,iv)
                pd.DataFrame({"row":iv,"actual":train["count"].to_numpy()[iv],"prediction":p,"signature":signature}).to_csv(path,index=False)
                log_event({"phase":"development","fold":month,"model":name,"rmsle":score(train["count"].to_numpy()[iv],p),
                           "train_rows":len(it),"validation_rows":len(iv),"seconds":round(time.monotonic()-start,2)})
            rows.append({"fold":month,"model":name,"rmsle":score(train["count"].to_numpy()[iv],p),"n_train":len(it),"n_valid":len(iv)})
        pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv",index=False)
    alliv = np.concatenate(pieces)
    truth = train["count"].to_numpy()[alliv]
    names = ["baseline"] + list(CONFIGS)
    preds = {name:np.concatenate([pd.read_csv(pred_dir / f"{m}_{name}.csv").prediction.to_numpy() for m in FOLDS]) for name in names}
    summary = pd.DataFrame([{"model":name,"pooled_rmsle":score(truth,preds[name]),
                           "mean_fold_rmsle":np.mean([r["rmsle"] for r in rows if r["model"]==name])} for name in names]).sort_values("pooled_rmsle")
    summary.to_csv(OUT / "model_comparison.csv",index=False)
    selected = summary.loc[summary.model.ne("baseline"),"model"].head(4).tolist()
    logs = np.column_stack([np.log1p(preds[n]) for n in selected])
    target = np.log1p(truth)
    opt = minimize(lambda w:np.mean((logs@w-target)**2),np.ones(len(selected))/len(selected),
                   method="SLSQP",bounds=[(0,1)]*len(selected),
                   constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                   options={"ftol":1e-12,"maxiter":1000})
    if not opt.success:
        raise RuntimeError(opt.message)
    weights = opt.x
    weights[weights<1e-6] = 0
    weights /= weights.sum()
    blended_log = logs@weights
    blend_score = score(truth,np.expm1(blended_log))
    # Check blend flexibility with leave-one-period-out weight fitting.
    cross_scores = []
    for j,month in enumerate(FOLDS):
        held = np.concatenate([np.full(len(v),k==j) for k,v in enumerate(pieces)])
        fitted = minimize(lambda w:np.mean((logs[~held]@w-target[~held])**2),
                         np.ones(len(selected))/len(selected),method="SLSQP",bounds=[(0,1)]*len(selected),
                         constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"ftol":1e-12})
        cross_scores.append({"fold":month,"blend_rmsle":float(np.sqrt(np.mean((logs[held]@fitted.x-target[held])**2)))})
    pd.DataFrame(cross_scores).to_csv(OUT / "blend_crosscheck.csv",index=False)
    # Freeze selection before opening holdout targets.
    best_single = selected[0]
    selection = {"models":selected,"weights":weights.tolist(),"development_rmsle":blend_score,
                 "best_single":best_single,"best_single_development_rmsle":score(truth,preds[best_single]),
                 "blend_leave_one_fold_out_mean":float(np.mean([r["blend_rmsle"] for r in cross_scores])),
                 "method":"nonnegative convex blend of log1p predictions; top 4 development models"}
    dump(OUT / "selection.json",selection)
    log_event({"phase":"selection_frozen",**selection})
    it,iv = split(dtrain,HOLDOUT)
    hold = {}
    for name,w in zip(selected,weights):
        if w>0 or name==best_single:
            hold[name],_ = candidate(name,x,groups,train,it,iv)
    hp = np.expm1(sum(w*np.log1p(hold[n]) for n,w in zip(selected,weights) if w>0))
    hb = baseline(train,x,it,iv)
    hold_metrics = {"fold":HOLDOUT,"train_rows":len(it),"validation_rows":len(iv),
                    "baseline_rmsle":score(train["count"].to_numpy()[iv],hb),
                    "best_single_rmsle":score(train["count"].to_numpy()[iv],hold[best_single]),
                    "blend_rmsle":score(train["count"].to_numpy()[iv],hp)}
    dump(OUT / "holdout_metrics.json",hold_metrics)
    log_event({"phase":"holdout_once",**hold_metrics})
    holdframe = pd.DataFrame({"row":iv,"datetime":train.datetime.iloc[iv].to_numpy(),"actual":train["count"].to_numpy()[iv],"prediction":hp})
    holdframe.to_csv(OUT / "holdout_predictions.csv",index=False)
    errors = []
    for dimension in ["hr","workingday","weathersit","weekday"]:
        hf = holdframe.copy(); hf[dimension] = x.iloc[iv][dimension].to_numpy().astype(int)
        for value,g in hf.groupby(dimension):
            errors.append({"dimension":dimension,"value":value,"n":len(g),"rmsle":score(g.actual,g.prediction)})
    pd.DataFrame(errors).to_csv(OUT / "holdout_error_slices.csv",index=False)
    # Full refit uses all provided training labels after the holdout estimate is recorded.
    (OUT / "models").mkdir(exist_ok=True)
    fullit = np.arange(len(train)); ip = np.arange(len(train),len(x))
    final = np.zeros(len(test)); imps = []
    individual = pd.DataFrame({"id":np.arange(1,len(test)+1)})
    for name,w in zip(selected,weights):
        if w<=0:
            continue
        start = time.monotonic()
        p,imp = candidate(name,x,groups,train,fullit,ip,save=True)
        individual[name] = p
        imps.append(imp); final += w*np.log1p(p)
        log_event({"phase":"full_refit","model":name,"weight":float(w),"seconds":round(time.monotonic()-start,2)})
    p = np.maximum(np.expm1(final),0)
    sub = pd.DataFrame({"id":np.arange(1,len(test)+1),"cnt":p})
    assert sub.columns.tolist()==["id","cnt"]
    assert len(sub)==len(test) and sub.id.is_unique
    assert np.isfinite(p).all() and (p>=0).all()
    sub.to_csv(ROOT / "submit.csv",index=False,float_format="%.8f")
    individual.to_csv(OUT / "test_model_predictions.csv",index=False)
    pd.concat(imps).to_csv(OUT / "feature_importance.csv",index=False)
    dump(OUT / "feature_schema.json",{"groups":groups,"categories":CATEGORIES,"configs":CONFIGS,"seed":SEED})
    oof = pd.DataFrame({"row":alliv,"datetime":train.datetime.iloc[alliv].to_numpy(),"actual":truth,
                        "prediction":np.expm1(blended_log)})
    oof.to_csv(OUT / "development_predictions.csv",index=False)
    fig,ax = plt.subplots(figsize=(13,4))
    ax.plot(pd.to_datetime(holdframe.datetime),holdframe.actual,label="Actual",alpha=.7)
    ax.plot(pd.to_datetime(holdframe.datetime),holdframe.prediction,label="Prediction",alpha=.7)
    ax.legend(); ax.set(title="Untouched December holdout",ylabel="Count"); fig.tight_layout()
    fig.savefig(OUT / "holdout.png",dpi=150); plt.close(fig)
    summary_stats = {"rows":len(sub),"min":float(p.min()),"median":float(np.median(p)),
                     "mean":float(p.mean()),"max":float(p.max()),"finite":bool(np.isfinite(p).all()),
                     "submission_sha256":hashlib.sha256((ROOT / "submit.csv").read_bytes()).hexdigest()}
    dump(OUT / "submission_checks.json",summary_stats)
    with (ROOT / "REPORT.md").open("a",encoding="utf-8") as f:
        f.write(f"""
## 実験結果
{table(summary)}

### fold別結果
{table(pd.DataFrame(rows).pivot(index="model",columns="fold",values="rmsle").reset_index())}

- baselineは時刻×勤務日×月の過去log需要平均（未観測組合せは時刻×勤務日にフォールバック）。
- basic→calendar→fullの比較で暦・気象派生特徴の追加効果を確認。ライブラリ間・深さ・平滑化・利用者区分別学習を比較。
- component系はcasual/registeredを別モデルでlog1p学習し、元の件数へ戻して合算。これらは目的変数としてのみ使用。
- 開発スコアによる上位4候補のlog1p予測を非負・総和1で混合。重みは開発データのみで最適化。
- 選択モデルと重み: {json.dumps(dict(zip(selected,weights.tolist())),ensure_ascii=False)}。
- 開発混合RMSLE={blend_score:.6f}、単体最良={selection['best_single_development_rmsle']:.6f}。
- foldを1つ除いて重みを学習した交差確認のfold平均={selection['blend_leave_one_fold_out_mean']:.6f}。
- この交差確認は上位モデル選択自体をネストしていないため、独立評価とは扱わない。最終ホールドアウトが独立評価。

## 一度だけ行った最終ホールドアウト
{table(pd.DataFrame([hold_metrics]))}

- ホールドアウトの結果に応じたモデル・重み・ハイパーパラメータの変更は行っていない。
- 予測誤差の時刻別・勤務日別・気象別集計はartifacts/holdout_error_slices.csv。
![holdout](artifacts/holdout.png)

## 最終学習・提出ファイル
- 選択済みモデルを全10,886行で再学習し、testの原順序に対して予測。
- submit.csvはid,cntの2列、{len(sub):,}行。idは1〜{len(sub)}、欠損・無限値・負値なし。
- 予測件数の範囲: {p.min():.4f}〜{p.max():.4f}、平均={p.mean():.4f}。RMSLE目的のため整数丸めは行わない。
- 変換、EDA、特徴量生成、開発検証、選定、ホールドアウト、最終予測の全手順はanalysis.py。
- 実験ごとの予測・foldスコア・混合比・環境バージョン・モデル・重要度をartifactsに保存。
- 入力原本のSHA256はartifacts/environment.json。提出ファイルSHA256はartifacts/submission_checks.json。
- 同じ原本・同じ依存環境・同じseedで再実行可能。ライブラリ/OS差で微小な数値差が生じる場合がある。
- 実際のtest正解は未知であり、本番RMSLE・Kaggle順位は算出していない。

## 再実行
1. Python 3.12でpython -m pip install -r requirements.txtを実行。
2. python analysis.pyで全工程を再現。既存rawは上書きせず、形式変換を何度実行しても原本から再構成。
3. 開発予測キャッシュを使わず再学習する場合はpython analysis.py --fresh。
4. 形式変換とEDAのみならpython analysis.py --prepare-only。
""")
    print("COMPLETE "+json.dumps({"holdout":hold_metrics,"submission":summary_stats},ensure_ascii=False),flush=True)


def write_findings():
    """Interpret saved experiments without changing the selected model."""
    comparison = pd.read_csv(OUT / "model_comparison.csv").set_index("model")
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    holdout = json.loads((OUT / "holdout_metrics.json").read_text(encoding="utf-8"))
    raw_train = pd.read_csv(RAW / "train.csv")
    raw_test = pd.read_csv(RAW / "test.csv")
    dates = pd.to_datetime(raw_train.datetime)
    shift = pd.DataFrame({
        "feature": ["temp", "atemp", "humidity", "windspeed", "holiday", "workingday"],
        "train_mean": raw_train[["temp","atemp","humidity","windspeed","holiday","workingday"]].mean().to_numpy(),
        "test_mean": raw_test[["temp","atemp","humidity","windspeed","holiday","workingday"]].mean().to_numpy()
    })
    shift.to_csv(OUT / "eda_split_means.csv", index=False)
    inconsistent = int(raw_train.workingday.ne((dates.dt.weekday.lt(5) & raw_train.holiday.eq(0)).astype(int)).sum())
    anomaly = raw_train.loc[(raw_train.temp/41 - raw_train.atemp/50).abs().nlargest(5).index,
                            ["datetime","temp","atemp","humidity","weather"]]
    anomaly.to_csv(OUT / "eda_feels_like_extremes.csv", index=False)
    paired = []
    for description, before, after in [
        ("LightGBM: basicから暦特徴追加", "lgb_basic", "lgb_calendar"),
        ("LightGBM: 暦から気象派生追加", "lgb_calendar", "lgb_weather"),
        ("CatBoost: 暦から気象派生追加", "cat_calendar", "cat_weather"),
        ("CatBoost: 総数直接から区分別合算", "cat_weather", "cat_components")
    ]:
        a, b = comparison.loc[before,"pooled_rmsle"], comparison.loc[after,"pooled_rmsle"]
        paired.append({"comparison":description,"before_rmsle":a,"after_rmsle":b,"delta_after_minus_before":b-a})
    imp = pd.read_csv(OUT / "feature_importance.csv")
    tops = []
    for (model, target), g in imp.groupby(["model","target"]):
        top = g.nlargest(8,"importance")
        tops.append({"model":model,"target":target,"top_features":", ".join(top.feature)})
    baseline_score = comparison.loc["baseline","pooled_rmsle"]
    development_improvement = 100*(1-selection["development_rmsle"]/baseline_score)
    holdout_improvement = 100*(1-holdout["blend_rmsle"]/holdout["baseline_rmsle"])
    marker = "\n## 結果の解釈と追加診断\n"
    report_path = ROOT / "REPORT.md"
    report = report_path.read_text(encoding="utf-8").split(marker)[0]
    report += marker + f"""
- 開発4期間の混合モデルRMSLEは{selection['development_rmsle']:.6f}。過去平均ベースライン比で{development_improvement:.1f}%改善。
- 未使用だった12月ホールドアウトのRMSLEは{holdout['blend_rmsle']:.6f}、ベースライン比で{holdout_improvement:.1f}%改善。
- 両者は別期間の指標であり、開発とホールドアウトの大小をそのまま性能改善とは扱わない。

### 同一設定内の比較
差分は負なら改善、正なら悪化。開発4期間を結合したRMSLEで比較。
{table(pd.DataFrame(paired))}

- 特徴量の追加は常に改善するわけではない。上表の実測値に従って最終候補を選定した。
- モデル重要度は各ライブラリの定義が異なり、モデル間の絶対値比較や因果解釈には使わない。
{table(pd.DataFrame(tops))}

### EDAで確認した注意点
- workingdayと「平日かつholiday=0」の不整合は{inconsistent}行。
- 勤務日は8時・17〜18時、非勤務日は昼過ぎに需要が集中。勤務日×時刻の交互作用を追加する根拠となった。
- 湿度ゼロは{int(raw_train.humidity.eq(0).sum())}行。真の観測か異常かは確定できないため、元の値を保持してゼロ指標を追加。
- 2012-08-17には高温にもかかわらず体感温度が低い記録がある。外部の根拠なく値を修正せず、気温と体感温度の差を特徴量にした。
{table(anomaly)}

### train/testの気象・暦の平均値
{table(shift)}

### 適用上の前提と残る不確実性
- 開発・最終ホールドアウトでは評価開始以降の実績を学習に使わない。
- 最終提出は配布されたtrain全体を既知の教師データとして再学習するコンペ形式。各test時点までの行だけで逐次学習するオンライン運用のシミュレーションとは異なる。
- 毎月20日以降の正解がないため、月末そのものやクリスマス期の精度は検証できない。月内の日にちそのものは特徴量から除外した。
- 今回の開発期間は2011-10以降。特に履歴の短い2011年初期への精度は独立には測定していない。
- 気象ラグは欠けた時間があれば欠損のまま扱う。全体を先読みして補間しない。
- 検証用気象・testの説明変数は提供済みとする。実運用で当該時刻の気象が未観測なら気象予報に置換した再評価が必要。
- 再現チェックはpython verify_outputs.py --models。保存モデルからの予測、混合比、submit.csvの一致も確認できる。
"""
    report_path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--fresh",action="store_true")
    args = parser.parse_args()
    if args.fresh:
        cache = OUT / "validation_predictions"
        # Only remove the known prediction CSVs inside the fixed artifact folder.
        for month in FOLDS:
            for name in ["baseline"]+list(CONFIGS):
                (cache / f"{month}_{name}.csv").unlink(missing_ok=True)
    train,test = prepare()
    eda(train,test)
    print("Prepared data and EDA. See REPORT.md.",flush=True)
    if not args.prepare_only:
        run_experiments(train,test)
        write_findings()

if __name__ == "__main__":
    main()

