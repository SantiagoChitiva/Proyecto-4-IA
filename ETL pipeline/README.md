# Pipeline ETL 
## Predicción de Cuartos de Final · Copa Mundial FIFA

---

## ¿Qué hace este pipeline?

Toma el archivo `train.csv` original (192 equipos de los mundiales 2002–2022) y lo deja listo para que los modelos de Random Forest, SVM y MLP puedan entrenarse. El script realiza limpieza, transformación y escalamiento, y genera 6 archivos de salida listos para usar.

---

## Requisitos

```
Python 3.8 o superior
pandas
scikit-learn
numpy
```

Instalación de dependencias:

```bash
pip install pandas scikit-learn numpy
```

---

## Cómo ejecutar

1. Colocar `train.csv` y `etl_pipeline.py` en la misma carpeta.
2. Ejecutar:

```bash
python etl_pipeline.py
```

3. Se generan los 6 archivos de salida en la misma carpeta.

---

## Archivos de entrada y salida

### Entrada

| Archivo | Descripción |
|---|---|
| `train.csv` | Dataset original. 192 filas × 24 columnas. Mundiales FIFA 2002–2022. |

### Salida

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `X_train.csv` | 153 | 23 | Features de entrenamiento **escaladas** → usar con SVM y MLP |
| `X_test.csv` | 39 | 23 | Features de prueba **escaladas** → usar con SVM y MLP |
| `X_train_raw.csv` | 153 | 23 | Features de entrenamiento **sin escalar** → usar con RF |
| `X_test_raw.csv` | 39 | 23 | Features de prueba **sin escalar** → usar con RF |
| `y_train.csv` | 153 | 1 | Etiquetas de entrenamiento (0 o 1) → usar con todos los modelos |
| `y_test.csv` | 39 | 1 | Etiquetas de prueba (0 o 1) → usar con todos los modelos |

### ¿Por qué X e y?

- **X** contiene las variables de entrada: todo lo que se sabe del equipo *antes* del mundial (ranking, goles, valor de mercado, etc.).
- **y** contiene la variable a predecir: `quarter_finalist` (0 = no clasificó a cuartos, 1 = sí clasificó).

### ¿Por qué hay versión escalada y sin escalar?

- **Random Forest** toma decisiones por comparaciones (`¿ranking > 20?`), no por distancias matemáticas. La escala no le afecta, puede usar los datos crudos.
- **SVM y MLP** son matemáticamente sensibles a la escala. Si una variable va de 1 a 105 y otra de €6M a €1.6B, la segunda domina el cálculo. Necesitan los datos escalados.

### ¿Qué archivo usa cada modelo?

| Modelo | X de entrenamiento | X de prueba | y |
|---|---|---|---|
| Random Forest | `X_train_raw.csv` | `X_test_raw.csv` | `y_train.csv` / `y_test.csv` |
| SVM | `X_train.csv` | `X_test.csv` | `y_train.csv` / `y_test.csv` |
| MLP | `X_train.csv` | `X_test.csv` | `y_train.csv` / `y_test.csv` |

---

## Pasos del pipeline explicados

### Paso 1 — Inspección inicial

Se revisan los tipos de variables, rangos mínimo/máximo y resumen estadístico de las 24 columnas originales. Se identifican:

- 19 variables numéricas
- 2 variables categóricas: `team` y `continent`
- 4 variables objetivo posibles: `quarter_finalist`, `semi_finalist`, `finalist`, `winner`

**Variable objetivo elegida:** `quarter_finalist`
**Justificación:** es la más balanceada (25% positivos vs 75% negativos). Las otras son demasiado desbalanceadas: `winner` solo tiene 3% de positivos, lo que haría muy difícil el aprendizaje.

---

### Paso 2 — Valores faltantes

**Columna afectada:** `squad_total_market_value_eur` — 32 valores nulos.

**Diagnóstico:** todos los nulos pertenecen al Mundial 2002. En ese año Transfermarkt no existía y no había registro del valor de mercado de las plantillas.

**Decisión:** imputar con la **mediana** de los 160 valores disponibles (mundiales 2006–2022).

**Valor imputado:** €217,940,000

**¿Por qué mediana y no media?**
La media quedó en €343M porque equipos con valores extremos como Inglaterra 2006 (€1.62B) la jalan hacia arriba. La mediana es el valor del punto medio de la distribución y es inmune a esos extremos, por lo que representa mejor a un equipo "típico".

**Resultado:** los 32 equipos del Mundial 2002 quedan todos con €217,940,000. Es una limitación conocida, Brasil y Arabia Saudita reciben el mismo valor, lo cual no es real, pero es la mejor estimación posible dado que no existe el dato histórico.

---

### Paso 3 — Eliminación de columnas innecesarias

| Columna eliminada | Motivo |
|---|---|
| `team` | Nombre del país. No aporta poder predictivo y puede causar sobreajuste (el modelo memorizaría nombres en lugar de aprender patrones). |
| `version` | Año del mundial. No es una característica del equipo; incluirlo haría que el modelo aprenda tendencias temporales en lugar de características reales. |
| `winner` | **Fuga de datos:** es un resultado del torneo. Si el modelo lo ve durante el entrenamiento, hace trampa, porque sabría quién ganó antes de predecir quién llegó a cuartos. |
| `finalist` | Fuga de datos |
| `semi_finalist` | Fuga de datos |

Quedan **19 columnas** luego de las eliminaciones.

---

### Paso 4 — Codificación de variables categóricas

**Variable afectada:** `continent` (6 categorías: Africa, Asia, Europe, North America, Oceania, South America).

**Decisión:** One-Hot Encoding (OHE).

**¿Por qué no simplemente asignar números (0, 1, 2...)?**
Asignar códigos numéricos implica una jerarquía falsa. Si Africa=0 y Europe=2, el modelo interpretaría que Europa "vale el doble" que África, lo cual no tiene sentido. OHE crea una columna binaria independiente por cada continente, sin ningún sesgo.

**Resultado:** `continent` (1 columna) se convierte en 6 columnas binarias:

| Columna generada | Valor si el equipo es de ese continente |
|---|---|
| `continent_Africa` | 1 |
| `continent_Asia` | 1 |
| `continent_Europe` | 1 |
| `continent_North America` | 1 |
| `continent_Oceania` | 1 |
| `continent_South America` | 1 |

`is_host` ya era binario (0 o 1), no requirió transformación.

El total de columnas sube de 19 a **23** (se reemplaza 1 columna por 6, neto +5, pero se eliminó `continent` original).

---

### Paso 5 — Desbalance de clases

| Clase | Cantidad | Porcentaje |
|---|---|---|
| 0 (no clasificó a cuartos) | 144 | 75% |
| 1 (sí clasificó a cuartos) | 48 | 25% |

El ratio es 3:1, considerado **desbalance moderado** (no extremo como 10:1 o 100:1).

**Implicaciones para los modelos:**

- Un modelo que siempre prediga 0 tendría 75% de accuracy sin aprender nada. Por eso no se debe usar accuracy como única métrica — usar F1-score.
- **RF y SVM:** usar `class_weight='balanced'` para que el modelo penalice más los errores en la clase minoritaria.
- **MLP:** monitorear con F1-score durante el entrenamiento.
- **Opcional:** si los resultados son pobres, aplicar SMOTE para generar ejemplos sintéticos de la clase minoritaria.

---

### Paso 6 — Train/Test Split

**División:** 80% entrenamiento / 20% prueba.

**Parámetro clave:** `stratify=quarter_finalist` garantiza que ambos conjuntos mantengan la misma proporción de clases (≈25% positivos en train y en test). Sin stratify, el split aleatorio podría concentrar todos los positivos en un solo conjunto.

| Conjunto | Filas | Positivos (clase 1) |
|---|---|---|
| Train | 153 | 38 (24.8%) |
| Test | 39 | 10 (25.6%) |

**`random_state=42`** fija la semilla aleatoria para que el split sea reproducible — cualquiera que ejecute el script obtendrá exactamente la misma división.

---

### Paso 7 — Escalamiento

**Método:** StandardScaler — transforma cada variable para que tenga media 0 y desviación estándar 1.

```
valor_escalado = (valor_original - media_entrenamiento) / std_entrenamiento
```

**Ejemplo con `goals_scored_last_4y`** (media ≈ 84, std ≈ 23):
- Un equipo con 60 goles → (60 - 84) / 23 = **-1.04** (está bajo la media)
- Un equipo con 110 goles → (110 - 84) / 23 = **+1.13** (está sobre la media)

Los valores negativos y decimales que aparecen en los CSV escalados son normales y esperados — no son errores.

**Columnas que SE escalan (16):** todas las variables numéricas continuas.

**Columnas que NO se escalan (7):** las 6 columnas OHE de continente e `is_host`, porque ya son binarias (0 o 1) y escalarlas no aporta nada.

**Regla crítica:** el scaler se ajusta (`fit`) **únicamente con X_train** y luego se aplica (`transform`) a X_test. Si se ajustara con el test set, se estaría filtrando información del futuro al modelo (fuga de datos).

---

## Variables del dataset final (23 features en X)

| Variable | Tipo | Descripción |
|---|---|---|
| `is_host` | Binaria | 1 si el equipo es anfitrión del mundial |
| `goals_scored_last_4y` | Numérica | Goles anotados en los últimos 4 años |
| `goals_received_last_4y` | Numérica | Goles recibidos en los últimos 4 años |
| `wins_last_4y` | Numérica | Victorias en los últimos 4 años |
| `losses_last_4y` | Numérica | Derrotas en los últimos 4 años |
| `draws_last_4y` | Numérica | Empates en los últimos 4 años |
| `world_cup_titles_before` | Numérica | Títulos mundiales previos |
| `squad_total_market_value_eur` | Numérica | Valor de mercado total de la plantilla en euros |
| `fifa_rank_pre_tournament` | Numérica | Ranking FIFA antes del torneo (1 = mejor) |
| `fifa_points_pre_tournament` | Numérica | Puntos FIFA antes del torneo |
| `squad_avg_age` | Numérica | Edad promedio de la plantilla |
| `world_cup_participations_before` | Numérica | Número de mundiales anteriores |
| `groups_passed_before` | Numérica | Veces que pasó la fase de grupos |
| `round16_before` | Numérica | Veces que llegó a octavos de final |
| `quarterfinals_before` | Numérica | Veces que llegó a cuartos de final |
| `semifinals_before` | Numérica | Veces que llegó a semifinales |
| `finals_before` | Numérica | Veces que llegó a la final |
| `continent_Africa` | Binaria (OHE) | 1 si el equipo es africano |
| `continent_Asia` | Binaria (OHE) | 1 si el equipo es asiático |
| `continent_Europe` | Binaria (OHE) | 1 si el equipo es europeo |
| `continent_North America` | Binaria (OHE) | 1 si el equipo es de Norteamérica |
| `continent_Oceania` | Binaria (OHE) | 1 si el equipo es de Oceanía |
| `continent_South America` | Binaria (OHE) | 1 si el equipo es suramericano |

**Variable objetivo (y):**

| Variable | Valores | Descripción |
|---|---|---|
| `quarter_finalist` | 0 / 1 | 1 = clasificó a cuartos de final |

---

## Resumen de decisiones tomadas

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| Imputar con mediana | Imputar con media / eliminar filas | Distribución asimétrica por outliers; eliminar filas perdería todo el Mundial 2002 |
| Eliminar `team` | Conservar como categórica | Demasiados valores únicos (62 países), causaría sobreajuste |
| Eliminar `version` | Conservar como numérica | El año no es una característica del equipo |
| One-Hot Encoding para `continent` | Label Encoding (0,1,2...) | Label Encoding implica jerarquía falsa entre continentes |
| StandardScaler | MinMaxScaler | Más robusto ante outliers presentes en el valor de mercado |
| stratify en split | Split aleatorio simple | Garantiza proporciones de clase iguales en train y test |

---

## Notas para los modelos

```python
# Cómo cargar los archivos en Python

import pandas as pd

# Para Random Forest
X_train = pd.read_csv('X_train_raw.csv')
X_test  = pd.read_csv('X_test_raw.csv')

# Para SVM y MLP
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')

# Para todos los modelos (mismo y)
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()
```

`.squeeze()` convierte el DataFrame de una columna en una Serie, que es el formato que esperan los modelos de scikit-learn.

---