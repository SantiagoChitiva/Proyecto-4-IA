import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Cargar datos
print("CARGA DE DATOS")

df = pd.read_csv("train.csv")
print(f"  Registros cargados : {df.shape[0]}")
print(f"  Columnas totales   : {df.shape[1]}")
print()

#Inspección inicial
print("INSPECCIÓN INICIAL")

#Tipos de variables
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

print(f"\n  Variables NUMÉRICAS ({len(num_cols)}):")
for c in num_cols:
    print(f"    {c:<40} min={df[c].min():.2f}  max={df[c].max():.2f}")

print(f"\n  Variables CATEGÓRICAS ({len(cat_cols)}):")
for c in cat_cols:
    print(f"    {c:<40} valores únicos: {df[c].nunique()}")

#Resumen estadístico
print("\n  Resumen estadístico (variables numéricas):")
print(df.describe().round(2).to_string())

#Variables objetivo disponibles
target_vars = ["quarter_finalist", "semi_finalist", "finalist", "winner"]
print("\n  Variables objetivo disponibles:")
for t in target_vars:
    counts = df[t].value_counts()
    pct_pos = counts.get(1, 0) / len(df) * 100
    print(f"    {t:<25}  0={counts.get(0,0)}  1={counts.get(1,0)}  ({pct_pos:.1f}% positivos)")

print("\n  Variable objetivo elegida: quarter_finalist")
print("    Porque es la más balanceada (25% positivos),")
print("    lo que permite un aprendizaje más justo para los modelos.")

# Análisis de valores faltantes
print("\n" + "VALORES FALTANTES")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("\n  No hay valores faltantes.")
else:
    print("\n  Columnas con nulos:")
    for col, n in missing.items():
        pct = n / len(df) * 100
        print(f"    {col:<40} {n} nulos ({pct:.1f}%)")

# Diagnóstico de squad_total_market_value_eur
col_mv = "squad_total_market_value_eur"
null_rows = df[df[col_mv].isnull()]
print(f"\n  Diagnóstico de '{col_mv}':")
print(f"    Los {len(null_rows)} nulos pertenecen al Mundial: "
      f"{null_rows['version'].unique()}")
print("    Son todos los equipos del Mundial 2002.")
print("    En 2002 no existía Transfermarkt; el dato no estaba disponible.")

mediana_val = df[col_mv].median()
media_val   = df[col_mv].mean()
print(f"\n  Media   del valor de mercado: €{media_val:,.0f}")
print(f"  Mediana del valor de mercado: €{mediana_val:,.0f}")
print("\n  Se decidio imputar con la MEDIANA")
print("    Porque la distribución del valor de mercado es asimétrica")
print("    (hay equipos con valores >€1B que jalan la media hacia arriba).")
print("    La mediana es más robusta ante esos valores extremos.")

# Aplicar imputación
df[col_mv] = df[col_mv].fillna(mediana_val)
print(f"\n  Nulos imputados con mediana = €{mediana_val:,.0f}")
print(f"  Nulos restantes: {df[col_mv].isnull().sum()}")

#Eliminación de columnas innecesarias
print("\n" +"ELIMINACIÓN DE COLUMNAS INNECESARIAS")

cols_to_drop = {
    "team"        : "Identificador del país. No aporta para la prediccion "
                    "y puede causar sobreajuste (memorizar nombres).",
    "version"     : "Año del mundial. No es una característica del equipo; ",
    "winner"      : "Variable objetivo derivada (más específica que quarter_finalist). "
                    "Incluirla causaría fuga de datos (data leakage).",
    "finalist"    : "Fuga de datos.",
    "semi_finalist": "Fuga de datos.",
}

print("\n  Columnas eliminadas y justificación:")
for col, reason in cols_to_drop.items():
    print(f"\n  ✗ {col}")
    print(f"    {reason}")

df = df.drop(columns=list(cols_to_drop.keys()))
print(f"\n  Columnas restantes: {df.shape[1]}  (eran {df.shape[1] + len(cols_to_drop)})")

#Transformación de variables categóricas
print("\n" +"CODIFICACIÓN DE VARIABLES CATEGÓRICAS")

print("\n  Variable categórica: 'continent'")
print(f"  Categorías únicas: {sorted(df['continent'].unique())}")
print("\n  Se decidio usar One-Hot Encoding (OHE).")
print("    Porque 'continent' NO tiene orden natural")
print("    Asignar numeros 0-5 seria una falsa jerarquía (Africa < Europe)")
print("    OHE crea una columna binaria por continente, sin sesgo")

# Aplicar OHE y eliminar la primera columna para evitar multicolinealidad
df = pd.get_dummies(df, columns=["continent"], drop_first=False, dtype=int)

continent_cols = [c for c in df.columns if c.startswith("continent_")]
print(f"\n  Columnas OHE generadas:")
for c in continent_cols:
    print(f"    {c}  →  {df[c].sum()} equipos")

print("\n  Nota: is_host ya es binario (0/1), no requiere transformación.")

#Revisión del desbalance de clases
print("\n" +"DESBALANCE DE CLASES")

target = "quarter_finalist"
counts = df[target].value_counts()
ratio  = counts[0] / counts[1]

print(f"\n  Distribución de '{target}':")
print(f"    Clase 0 (no clasificó): {counts[0]}  ({counts[0]/len(df)*100:.1f}%)")
print(f"    Clase 1 (sí clasificó): {counts[1]}  ({counts[1]/len(df)*100:.1f}%)")
print(f"    Ratio desbalance      : {ratio:.1f}:1")

print("\n  ANÁLISIS:")
print("    El desbalance 3:1 es moderado")

#Separación X / y y Train-Test Split
print("\n" + "SEPARACIÓN X / y  Y  TRAIN-TEST SPLIT")

y = df[target]
X = df.drop(columns=[target])

print(f"\n  X (features): {X.shape[1]} columnas × {X.shape[0]} filas")
print(f"  y (target)  : {y.shape[0]} valores")

print(f"\n  Features incluidas en X:")
for c in X.columns:
    print(f"    {c}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y           # garantiza mismo ratio en train y test
)

print(f"\n  Split 80/20 con stratify=quarter_finalist:")
print(f"    X_train: {X_train.shape[0]} filas  |  "
      f"positivos: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"    X_test : {X_test.shape[0]} filas  |  "
      f"positivos: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

#Escalamiento de variables
print("\n" +"ESCALAMIENTO (StandardScaler)")

# Solo escalar columnas numéricas continuas (no las binarias OHE ni is_host)
binary_cols = continent_cols + ["is_host"]
scale_cols  = [c for c in X_train.columns if c not in binary_cols]

print(f"\n  Columnas que SE escalan ({len(scale_cols)}):")
for c in scale_cols:
    print(f"    {c:<40}  "
          f"rango original: [{X_train[c].min():.1f}, {X_train[c].max():.1f}]")

print(f"\n  Columnas que NO se escalan (ya son binarias 0/1) ({len(binary_cols)}):")
for c in binary_cols:
    print(f"    {c}")

print("\n  DECISIÓN: StandardScaler (media 0, desviación estándar 1).")
print("    Porque SVM y MLP son sensibles a la escala.")
print("    El ranking FIFA va de 1 a 105, el valor de mercado va de")
print("    €6M a €1.6B, sin escalar, las variables grandes dominarían.")

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols]  = scaler.transform(X_test[scale_cols])

print("\n  Escalamiento aplicado. Verificación en X_train:")
for c in scale_cols[:3]:
    m = X_train_scaled[c].mean()
    s = X_train_scaled[c].std()
    print(f"    {c:<35}  media={m:.4f}  std={s:.4f}")
print("    ...")

#Guardar archivos
print("\n" +"GUARDANDO ARCHIVOS")

X_train_scaled.to_csv("X_train.csv", index=False)
X_test_scaled.to_csv("X_test.csv",   index=False)
y_train.to_csv("y_train.csv",        index=False)
y_test.to_csv("y_test.csv",          index=False)

# También guardar sin escalar (para RF que no lo necesita)
X_train.to_csv("X_train_raw.csv", index=False)
X_test.to_csv("X_test_raw.csv",   index=False)

print("\n  Archivos generados:")
print("    X_train.csv      → features entrenamiento (escaladas)  para SVM y MLP")
print("    X_test.csv       → features prueba        (escaladas)  para SVM y MLP")
print("    X_train_raw.csv  → features entrenamiento (sin escalar) para RF")
print("    X_test_raw.csv   → features prueba        (sin escalar) para RF")
print("    y_train.csv      → etiquetas entrenamiento")
print("    y_test.csv       → etiquetas prueba")

print(f"\n  Shape final:")
print(f"    X_train: {X_train_scaled.shape}  →  {X_train_scaled.shape[1]} features")
print(f"    X_test : {X_test_scaled.shape}")
print(f"    y_train: {y_train.shape}")
print(f"    y_test : {y_test.shape}")

# RESUMEN FINAL
print("\n" +"Pipeline ETL completado")