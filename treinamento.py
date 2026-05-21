import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

df = pd.read_csv('dataset/Earthquakes_USGS.csv')

features = ['latitude', 'longitude', 'depth', 'rms', 'gap', 'sig']

# remove dados nulos e linhas inconsistentes (ex: gap zerado)
df_clean = df[features + ['mag']].dropna()
df_clean = df_clean[df_clean['gap'] > 0]

X = df_clean[features]
y = df_clean['mag']

print(f"--- Diagnóstico Base USGS ---")
print(f"Total de registros limpos para treino: {len(df_clean)}")
print(f"Média histórica de Magnitude: {y.mean():.2f} Mw")

# Divisão (Treino e Teste 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define o threshold de risco de desastre (Magnitude >= 5.0 Richter)
THRESHOLD_ALERTA = 5.0
y_test_class = (y_test >= THRESHOLD_ALERTA).astype(int)

# Treina os Algoritmos (Regressão Linear e Random Forest)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

print(f"\nVerificação de Dinâmica das Predições:")
print(f"Regressão Linear -> Mín: {lr_preds.min():.2f} | Máx: {lr_preds.max():.2f}")
print(f"Random Forest    -> Mín: {rf_preds.min():.2f} | Máx: {rf_preds.max():.2f}")

# Cálculo das métricas de avaliação para ambos os modelos
def calcular_metricas(y_true_reg, y_pred_reg, y_true_class):
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    y_pred_class = (y_pred_reg >= THRESHOLD_ALERTA).astype(int)
    
    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
    
    return {
        "RMSE (Regressão)": rmse,
        "Acurácia": acc,
        "Precisão": prec,
        "Recall": rec,
        "F1-Score": f1
    }

metrics = {
    "Regressão Linear": calcular_metricas(y_test, lr_preds, y_test_class),
    "Random Forest": calcular_metricas(y_test, rf_preds, y_test_class)
}

# Exportação dos modelos e métricas para uso no Streamlit
os.makedirs('modelo', exist_ok=True)
with open('modelo/modelo_regressao_linear.pkl', 'wb') as f: pickle.dump(lr_model, f)
with open('modelo/modelo_random_forest.pkl', 'wb') as f: pickle.dump(rf_model, f)
with open('modelo/metricas.pkl', 'wb') as f: pickle.dump(metrics, f)

print("\n[SUCESSO] Modelos USGS prontos para o Streamlit!")