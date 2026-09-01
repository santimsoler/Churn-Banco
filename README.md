# Datos

## `raw/`

`churn_train.csv` (8.101 filas) y `churn_test.csv` (2.026 filas). Partición
provista por la cátedra sobre el dataset *Credit Card Customers*, publicado en
Kaggle:

<https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers>

10.127 clientes en total, 16,1% de abandono en ambas particiones.

### Columnas

| Columna | Descripción |
|---|---|
| `Churn` | Variable objetivo. 1 si el cliente se dio de baja |
| `Customer_Age` | Edad en años |
| `Gender` | Género (M/F) |
| `Dependent_count` | Personas a cargo |
| `Education_Level` | Nivel educativo |
| `Marital_Status` | Estado civil |
| `Income_Category` | Categoría de ingresos anuales, en tramos |
| `Card_Category` | Tipo de tarjeta (Blue, Silver, Gold, Platinum) |
| `Months_on_book` | Antigüedad como cliente, en meses |
| `Total_Relationship_Count` | Productos bancarios contratados |
| `Months_Inactive_12_mon` | Meses de inactividad en los últimos 12 |
| `Contacts_Count_12_mon` | Contactos con el banco en los últimos 12 meses |
| `Credit_Limit` | Límite de la tarjeta |
| `Total_Revolving_Bal` | Saldo rotativo |
| `Avg_Open_To_Buy` | Crédito disponible. **Descartada** (ver abajo) |
| `Total_Amt_Chng_Q4_Q1` | Variación de monto transaccionado, Q4 vs Q1 |
| `Total_Trans_Amt` | Monto total transaccionado en 12 meses |
| `Total_Trans_Ct` | Cantidad total de transacciones en 12 meses |
| `Total_Ct_Chng_Q4_Q1` | Variación de cantidad de transacciones, Q4 vs Q1 |
| `Avg_Utilization_Ratio` | Ratio de utilización de la tarjeta |

Los CSV traen además una columna `Unnamed: 0` que es el índice de fila del
archivo, no una variable. `src/data.py` la elimina en la carga.

### Columnas descartadas

`Avg_Open_To_Buy = Credit_Limit − Total_Revolving_Bal`, identidad exacta
(error absoluto máximo: 5,7e-14). Es una combinación lineal, no información
nueva: infla la varianza de los coeficientes en los modelos lineales y reparte
la importancia entre columnas redundantes en los de árboles.
`src/data.check_linear_dependencies` verifica la identidad.

## `processed/`

`scoring_holdout.csv` lo genera `scripts/run_pipeline.py`. Contiene el holdout
con cuatro columnas agregadas:

- `score`: probabilidad de abandono predicha
- `decil`: decil de riesgo, 1 = mayor riesgo
- `priorizado`: 1 si el score supera el umbral congelado
- `segmento`: cluster asignado dentro de la cohorte priorizada

## Advertencia sobre el dataset

Es un dataset sintético o fuertemente depurado. Resulta casi separable para
ensambles de árboles (ROC-AUC 0,99), lo que no ocurre con datos bancarios
reales. Ver la sección correspondiente del README principal.
