"""Independent artifact checks; run after python analysis.py.
Use --models to additionally regenerate predictions from persisted models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    args = parser.parse_args()
    out = ROOT / "artifacts"
    load = lambda n: json.loads((out / n).read_text(encoding="utf-8"))
    env = load("environment.json")
    norm = load("normalization.json")
    originals = {}
    expected = ["id","dteday","season","yr","mnth","hr","holiday","weekday","workingday",
                "weathersit","temp","atemp","hum","windspeed","cnt"]
    for name in ["train","test"]:
        raw_path = ROOT / "raw" / f"{name}.csv"
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == env["raw_sha256"][name]
        a = pd.read_csv(raw_path)
        originals[name] = a
        b = pd.read_csv(ROOT / f"{name}.csv")
        assert b.columns.tolist() == expected
        assert len(a) == len(b)
        np.testing.assert_array_equal(b.id,np.arange(1,len(b)+1))
        dt = pd.to_datetime(a.datetime)
        np.testing.assert_array_equal(b.weekday,(dt.dt.dayofweek+1)%7)
        np.testing.assert_array_equal(b.yr,dt.dt.year-2011)
        np.testing.assert_array_equal(b.mnth,dt.dt.month)
        np.testing.assert_array_equal(b.hr,dt.dt.hour)
        np.testing.assert_array_equal(b.dteday,dt.dt.strftime("%Y-%m-%d"))
        for dst,src in [("season","season"),("holiday","holiday"),("workingday","workingday"),
                        ("weathersit","weather"),("windspeed","windspeed")]:
            np.testing.assert_allclose(b[dst],a[src])
        for dst,src in [("temp","temp"),("atemp","atemp"),("hum","humidity")]:
            s = norm[name][dst]
            np.testing.assert_allclose(b[dst],(a[src]-s["min"])/(s["max"]-s["min"]),atol=1e-12)
            assert b[dst].between(0,1).all()
        if name=="train":
            np.testing.assert_array_equal(b.cnt,a["count"])
        else:
            assert b.cnt.isna().all()
    selection = load("selection.json")
    w = np.asarray(selection["weights"])
    assert np.isclose(w.sum(),1) and (w>=0).all()
    sub = pd.read_csv(ROOT / "submit.csv")
    assert sub.columns.tolist()==["id","cnt"]
    assert len(sub)==len(originals["test"])
    np.testing.assert_array_equal(sub.id,np.arange(1,len(sub)+1))
    assert np.isfinite(sub.cnt).all() and sub.cnt.ge(0).all()
    individual = pd.read_csv(out / "test_model_predictions.csv")
    reconstructed = np.expm1(sum(weight*np.log1p(individual[name].to_numpy())
                                 for name,weight in zip(selection["models"],w) if weight>0))
    np.testing.assert_allclose(sub.cnt,reconstructed,atol=5.1e-9,rtol=1e-12)
    checks = load("submission_checks.json")
    assert checks["submission_sha256"]==hashlib.sha256((ROOT / "submit.csv").read_bytes()).hexdigest()
    hp = pd.read_csv(out / "holdout_predictions.csv")
    actual = originals["train"]["count"].to_numpy()[hp.row]
    np.testing.assert_array_equal(actual,hp.actual)
    rmsle = np.sqrt(np.mean((np.log1p(hp.prediction)-np.log1p(actual))**2))
    assert np.isclose(rmsle,load("holdout_metrics.json")["blend_rmsle"])
    import analysis
    dt = pd.to_datetime(originals["train"].datetime)
    for month in analysis.FOLDS+[analysis.HOLDOUT]:
        it,iv = analysis.split(dt,month)
        assert dt.iloc[it].max()<dt.iloc[iv].min()
        assert dt.iloc[iv].dt.day.between(11,19).all()
    if args.models:
        import joblib
        x,groups,_ = analysis.features(originals["train"],originals["test"])
        for name,weight in zip(selection["models"],w):
            if weight<=0:
                continue
            cfg = analysis.CONFIGS[name]
            xx = x.iloc[len(originals["train"]):][groups[cfg["features"]]].copy()
            if cfg["engine"]=="cat":
                for col in analysis.CATEGORIES:
                    if col in xx:
                        xx[col] = xx[col].fillna(-1).astype(int)
            targets = ["casual","registered"] if cfg.get("components") else ["count"]
            pred = np.zeros(len(xx))
            for target in targets:
                model = joblib.load(out / "models" / f"{name}_{target}.joblib")
                pred += np.maximum(np.expm1(np.clip(model.predict(xx),-10,15)),0)
            np.testing.assert_allclose(pred,individual[name],atol=1e-9,rtol=1e-10)
    print(f"PASS: schema, normalization, raw hashes, temporal splits, holdout metric, blend, submission"
          f"{', persisted models' if args.models else ''}. {len(sub)} predictions; holdout RMSLE={rmsle:.6f}")

if __name__=="__main__":
    main()

