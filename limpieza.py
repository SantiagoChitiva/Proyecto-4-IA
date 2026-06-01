import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar
df = pd.read_csv("train.csv")

# Imputar nulos
mediana = df["squad_total_market_value_eur"].median()
df["squad_total_market_value_eur"] = df["squad_total_market_value_eur"].fillna(mediana)

# Eliminar columnas innecesarias
df = df.drop(columns=["team", "version", "winner", "finalist", "semi_finalist"])

# One-Hot Encoding
df = pd.get_dummies(df, columns=["continent"], drop_first=False, dtype=int)

# Escalar variables numéricas
binary_cols = [c for c in df.columns if c.startswith("continent_")] + ["is_host", "quarter_finalist"]
scale_cols  = [c for c in df.columns if c not in binary_cols]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Guardar UN solo archivo
df.to_csv("dataset_limpio.csv", index=False)
print("dataset_limpio.csv generado.")