"""Entrena dos clasificadores con GridSearchCV y registra experimentos en MLflow."""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_prep import obtener_datos_publicos, preparar_dataframe

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "resultados"; OUT.mkdir(exist_ok=True)
raw_path = BASE / "datos" / "datos_ini" / "wdbc_original.csv"
if raw_path.exists(): raw = pd.read_csv(raw_path)
else: raw = obtener_datos_publicos()
df = preparar_dataframe(raw)
feature_cols = [c for c in df.columns if c not in ("diagnostico", "target_maligno")]
X, y = df[feature_cols], df["target_maligno"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
models = {
    "Regresion_Logistica": (Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",LogisticRegression(max_iter=5000,random_state=42))]), {"model__C":[0.1,1.0,10.0],"model__class_weight":[None,"balanced"]}),
    "Random_Forest": (Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(random_state=42,n_jobs=-1))]), {"model__n_estimators":[100,200],"model__max_depth":[None,8],"model__min_samples_leaf":[1,2]})
}
mlflow.set_tracking_uri((BASE / "mlruns").resolve().as_uri())
mlflow.set_experiment("Actividad 5 - Comparacion de modelos")
all_rows=[]; fold_scores={}
for name,(pipe,grid) in models.items():
    search = GridSearchCV(pipe,grid,scoring={"accuracy":"accuracy","precision_macro":"precision_macro","recall_macro":"recall_macro","f1_macro":"f1_macro","roc_auc":"roc_auc"},refit="f1_macro",cv=cv,n_jobs=-1,return_train_score=True)
    search.fit(X_train,y_train)
    model=search.best_estimator_; pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,list(model.classes_).index(1)]
    metrics={"accuracy":accuracy_score(y_test,pred),"precision_macro":precision_score(y_test,pred,average="macro",zero_division=0),"recall_macro":recall_score(y_test,pred,average="macro",zero_division=0),"f1_macro":f1_score(y_test,pred,average="macro",zero_division=0),"roc_auc":roc_auc_score(y_test,prob),"cv_f1_macro_media":search.best_score_,"cv_f1_macro_desv":search.cv_results_["std_test_f1_macro"][search.best_index_]}
    start=time.perf_counter()
    for _ in range(20): model.predict(X_test)
    latency_ms_per_row=(time.perf_counter()-start)/20*1000/len(X_test)
    metrics["latencia_ms_por_fila"]=latency_ms_per_row
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"dataset":"UCI WDBC","tipo_proyecto":"clasificacion academica","split":"80/20 estratificado","cv":"5-fold stratified","seed":42})
        mlflow.log_params({"modelo":name,"cv_folds":5,"test_size":0.2,"random_state":42, **{k:str(v) for k,v in search.best_params_.items()}})
        mlflow.log_metrics({k:float(v) for k,v in metrics.items()})
        mlflow.log_metric("grid_combinaciones",len(search.cv_results_["params"]))
        mlflow.sklearn.log_model(model,artifact_path="modelo")
        mlflow.log_artifact(str(BASE / "datos" / "datos_limp" / "wdbc_limpio.csv"), artifact_path="datos")
        mlflow.log_artifact(str(BASE / "datos" / "datos_limp" / "wdbc_limpio.csv"), artifact_path="datos")
        all_rows.append({"modelo":name,"run_id":run.info.run_id,**metrics,"mejores_hiperparametros":json.dumps(search.best_params_,ensure_ascii=False)})
        fold_scores[name]=[float(search.cv_results_[f"split{i}_test_f1_macro"][search.best_index_]) for i in range(5)]
    pd.DataFrame({"y_real":y_test.values,"y_predicho":pred,"prob_maligno":prob}).to_csv(OUT/f"predicciones_{name}.csv",index=False)
    pd.DataFrame(search.cv_results_).to_csv(OUT/f"gridsearch_{name}.csv",index=False)
    with open(OUT/f"reporte_{name}.txt","w",encoding="utf-8") as f: f.write(classification_report(y_test,pred,target_names=["benigno","maligno"]))

results=pd.DataFrame(all_rows)
results.to_csv(OUT/"metricas_modelos.csv",index=False)
try:
    from scipy.stats import wilcoxon
    stat,p=wilcoxon(fold_scores["Regresion_Logistica"],fold_scores["Random_Forest"],alternative="two-sided")
    comparison={"prueba":"Wilcoxon pareada sobre F1 de 5 folds (exploratoria)","estadistico":float(stat),"p_value":float(p),"scores":fold_scores,"limitacion":"Cinco folds y selección de hiperparámetros: resultado exploratorio, no evidencia concluyente."}
except Exception as e: comparison={"prueba":"No disponible","error":str(e),"scores":fold_scores}
(OUT/"comparacion_estadistica.json").write_text(json.dumps(comparison,indent=2,ensure_ascii=False),encoding="utf-8")
print(results.to_string(index=False))
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("Para revisar la interfaz: mlflow ui --backend-store-uri", (BASE/"mlruns").resolve().as_uri(), "--host 127.0.0.1 --port 5000")
