import pandas as pd
import random
import math
import os


# ============================================================
# 1. CARGAR LOS DATOS
# ============================================================

ruta = os.path.join(
    os.path.dirname(__file__),
    "loan_data.csv"
)

df = pd.read_csv(ruta)

print("Columnas del dataset:")
print(df.columns)


# ============================================================
# 2. FILTRAR PERSONAS DE 40 A 45 AÑOS
# ============================================================

df = df[
    (df["person_age"] >= 40) &
    (df["person_age"] <= 45)
]

print("\nCantidad de registros entre 40 y 45 años:")
print(len(df))


# ============================================================
# 3. SELECCIONAR ATRIBUTOS CATEGÓRICOS
# ============================================================

atributos = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

objetivo = "loan_status"


# ============================================================
# 4. SEPARAR X E Y
# ============================================================

X = df[atributos]
y = df[objetivo]


# ============================================================
# 5. DIVIDIR EN 80% ENTRENAMIENTO Y 20% PRUEBA
# ============================================================

indices = list(df.index)

random.seed(42)
random.shuffle(indices)

cantidad_entrenamiento = int(len(indices) * 0.80)

indices_entrenamiento = indices[:cantidad_entrenamiento]
indices_prueba = indices[cantidad_entrenamiento:]

X_train = X.loc[indices_entrenamiento]
y_train = y.loc[indices_entrenamiento]

X_test = X.loc[indices_prueba]
y_test = y.loc[indices_prueba]

print("\nCantidad total:", len(df))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))


# ============================================================
# EJERCICIO 1
# ENTROPÍA DE SHANNON
# ============================================================

def entropia(y):

    cantidad_0 = 0
    cantidad_1 = 0

    for valor in y:

        if valor == 0:
            cantidad_0 += 1
        else:
            cantidad_1 += 1

    total = len(y)

    p0 = cantidad_0 / total
    p1 = cantidad_1 / total

    resultado = 0

    if p0 > 0:
        resultado -= p0 * math.log2(p0)

    if p1 > 0:
        resultado -= p1 * math.log2(p1)

    return resultado


# ============================================================
# GANANCIA DE INFORMACIÓN
# ============================================================

def ganancia_informacion(X, y, atributo):

    entropia_inicial = entropia(y)

    valores = X[atributo].unique()

    entropia_ponderada = 0

    for valor in valores:

        indices = X[atributo] == valor

        y_subconjunto = y[indices]

        proporcion = len(y_subconjunto) / len(y)

        entropia_grupo = entropia(y_subconjunto)

        entropia_ponderada += (
            proporcion * entropia_grupo
        )

    ganancia = (
        entropia_inicial -
        entropia_ponderada
    )

    return ganancia


# ============================================================
# MOSTRAR GANANCIA
# ============================================================

print("\nGanancia de información:")

for atributo in atributos:

    ganancia = ganancia_informacion(
        X_train,
        y_train,
        atributo
    )

    print(atributo, "->", ganancia)


# ============================================================
# ID3
# ============================================================

def id3(X, y, atributos):

    if len(y.unique()) == 1:
        return int(y.iloc[0])

    if len(atributos) == 0:
        return int(y.mode()[0])

    mejor_atributo = None
    mayor_ganancia = -1

    for atributo in atributos:

        ganancia = ganancia_informacion(
            X,
            y,
            atributo
        )

        if ganancia > mayor_ganancia:
            mayor_ganancia = ganancia
            mejor_atributo = atributo

    if mayor_ganancia <= 0:
        return int(y.mode()[0])

    arbol = {
        mejor_atributo: {}
    }

    atributos_restantes = [
        atributo
        for atributo in atributos
        if atributo != mejor_atributo
    ]

    valores = X[mejor_atributo].unique()

    for valor in valores:

        indices = X[mejor_atributo] == valor

        X_sub = X[indices]
        y_sub = y[indices]

        arbol[mejor_atributo][valor] = id3(
            X_sub,
            y_sub,
            atributos_restantes
        )

    return arbol


# ============================================================
# CONSTRUIR ÁRBOL ID3
# ============================================================

arbol = id3(
    X_train,
    y_train,
    atributos
)


# ============================================================
# MOSTRAR ÁRBOL
# ============================================================

def imprimir_arbol(arbol, nivel=0):

    espacios = "    " * nivel

    if not isinstance(arbol, dict):

        print(espacios + "→ " + str(arbol))

        return

    atributo = next(iter(arbol))

    print(espacios + atributo)

    ramas = arbol[atributo]

    for valor, subarbol in ramas.items():

        print(
            espacios +
            "├── " +
            str(valor)
        )

        imprimir_arbol(
            subarbol,
            nivel + 1
        )


print("\n========================================")
print("ÁRBOL ID3")
print("========================================")

imprimir_arbol(arbol)


# ============================================================
# PREDICCIÓN ID3
# ============================================================

def predecir(arbol, ejemplo):

    if not isinstance(arbol, dict):
        return arbol

    atributo = next(iter(arbol))

    valor = ejemplo[atributo]

    if valor in arbol[atributo]:

        return predecir(
            arbol[atributo][valor],
            ejemplo
        )

    return 0


# ============================================================
# PREDICCIONES ID3
# ============================================================

predicciones = []

for indice in X_test.index:

    ejemplo = X_test.loc[indice]

    prediccion = predecir(
        arbol,
        ejemplo
    )

    predicciones.append(prediccion)


# ============================================================
# MATRIZ DE CONFUSIÓN ID3
# ============================================================

VP = 0
VN = 0
FP = 0
FN = 0

for real, predicho in zip(y_test, predicciones):

    if real == 1 and predicho == 1:
        VP += 1

    elif real == 0 and predicho == 0:
        VN += 1

    elif real == 0 and predicho == 1:
        FP += 1

    elif real == 1 and predicho == 0:
        FN += 1


print("\n========================================")
print("MATRIZ DE CONFUSIÓN - ID3")
print("========================================")

print("                 Predicho")
print("              0          1")
print()
print("Real  0      ", VN, "       ", FP)
print("      1      ", FN, "       ", VP)


# ============================================================
# MÉTRICAS ID3
# ============================================================

total = VP + VN + FP + FN

accuracy = (VP + VN) / total

if VP + FP > 0:
    precision = VP / (VP + FP)
else:
    precision = 0

if VP + FN > 0:
    recall = VP / (VP + FN)
else:
    recall = 0

if precision + recall > 0:
    f1 = (
        2 *
        (precision * recall) /
        (precision + recall)
    )
else:
    f1 = 0


print("\n========================================")
print("MÉTRICAS - ID3")
print("========================================")

print("VP:", VP)
print("VN:", VN)
print("FP:", FP)
print("FN:", FN)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")


# ============================================================
# EJERCICIO 2
# RANDOM FOREST
# ============================================================


# ============================================================
# CONSTRUIR ÁRBOL ALEATORIO
# ============================================================

def id3_random(
    X,
    y,
    atributos,
    max_profundidad,
    profundidad=0
):

    if len(y.unique()) == 1:
        return int(y.iloc[0])

    if len(atributos) == 0:
        return int(y.mode()[0])

    if profundidad >= max_profundidad:
        return int(y.mode()[0])

    # Seleccionar aleatoriamente atributos

    cantidad_atributos = max(
        1,
        int(math.sqrt(len(atributos)))
    )

    atributos_disponibles = atributos.copy()

    random.shuffle(atributos_disponibles)

    atributos_seleccionados = atributos_disponibles[
        :cantidad_atributos
    ]

    # Buscar mejor atributo

    mejor_atributo = None
    mayor_ganancia = -1

    for atributo in atributos_seleccionados:

        ganancia = ganancia_informacion(
            X,
            y,
            atributo
        )

        if ganancia > mayor_ganancia:

            mayor_ganancia = ganancia
            mejor_atributo = atributo

    if mayor_ganancia <= 0:
        return int(y.mode()[0])

    arbol = {
        mejor_atributo: {}
    }

    atributos_restantes = [
        atributo
        for atributo in atributos
        if atributo != mejor_atributo
    ]

    valores = X[mejor_atributo].unique()

    for valor in valores:

        indices = X[mejor_atributo] == valor

        X_sub = X[indices]
        y_sub = y[indices]

        arbol[mejor_atributo][valor] = id3_random(
            X_sub,
            y_sub,
            atributos_restantes,
            max_profundidad,
            profundidad + 1
        )

    return arbol


# ============================================================
# CREAR MUESTRA ALEATORIA
# ============================================================

def crear_muestra(X, y):

    indices = list(X.index)

    indices_muestra = random.choices(
        indices,
        k=len(indices)
    )

    X_muestra = X.loc[indices_muestra]
    y_muestra = y.loc[indices_muestra]

    return X_muestra, y_muestra


# ============================================================
# CONSTRUIR RANDOM FOREST
# ============================================================

def construir_random_forest(
    X,
    y,
    atributos,
    cantidad_arboles,
    max_profundidad
):

    bosque = []

    for i in range(cantidad_arboles):

        X_muestra, y_muestra = crear_muestra(
            X,
            y
        )

        arbol = id3_random(
            X_muestra,
            y_muestra,
            atributos,
            max_profundidad
        )

        bosque.append(arbol)

    return bosque


# ============================================================
# PREDICCIÓN DE UN ÁRBOL
# ============================================================

def predecir_arbol(arbol, ejemplo):

    if not isinstance(arbol, dict):
        return arbol

    atributo = next(iter(arbol))

    valor = ejemplo[atributo]

    if valor in arbol[atributo]:

        return predecir_arbol(
            arbol[atributo][valor],
            ejemplo
        )

    return 0


# ============================================================
# PREDICCIÓN RANDOM FOREST
# ============================================================

def predecir_random_forest(bosque, X):

    predicciones = []

    for indice in X.index:

        ejemplo = X.loc[indice]

        votos_0 = 0
        votos_1 = 0

        for arbol in bosque:

            prediccion = predecir_arbol(
                arbol,
                ejemplo
            )

            if prediccion == 0:
                votos_0 += 1
            else:
                votos_1 += 1

        if votos_1 > votos_0:
            predicciones.append(1)
        else:
            predicciones.append(0)

    return predicciones


# ============================================================
# RANDOM FOREST CON 10 ÁRBOLES
# ============================================================

random.seed(42)

cantidad_arboles = 10
max_profundidad = 5

bosque = construir_random_forest(
    X_train,
    y_train,
    atributos,
    cantidad_arboles,
    max_profundidad
)

print("\n========================================")
print("RANDOM FOREST")
print("========================================")

print("Cantidad de árboles:", cantidad_arboles)
print("Profundidad máxima:", max_profundidad)


# ============================================================
# PREDICCIONES RANDOM FOREST
# ============================================================

predicciones_rf = predecir_random_forest(
    bosque,
    X_test
)

print("\nCantidad de predicciones:", len(predicciones_rf))

print("\nPrimeras predicciones:")
print(predicciones_rf[:20])


# ============================================================
# MATRIZ DE CONFUSIÓN RANDOM FOREST
# ============================================================

VP = 0
VN = 0
FP = 0
FN = 0

for real, predicho in zip(y_test, predicciones_rf):

    if real == 1 and predicho == 1:
        VP += 1

    elif real == 0 and predicho == 0:
        VN += 1

    elif real == 0 and predicho == 1:
        FP += 1

    elif real == 1 and predicho == 0:
        FN += 1


print("\n========================================")
print("MATRIZ DE CONFUSIÓN - RANDOM FOREST")
print("========================================")

print("                 Predicho")
print("              0          1")
print()
print("Real  0      ", VN, "       ", FP)
print("      1      ", FN, "       ", VP)


# ============================================================
# MÉTRICAS RANDOM FOREST
# ============================================================

total = VP + VN + FP + FN

accuracy_rf = (VP + VN) / total

if VP + FP > 0:
    precision_rf = VP / (VP + FP)
else:
    precision_rf = 0

if VP + FN > 0:
    recall_rf = VP / (VP + FN)
else:
    recall_rf = 0

if precision_rf + recall_rf > 0:

    f1_rf = (
        2 *
        (precision_rf * recall_rf) /
        (precision_rf + recall_rf)
    )

else:
    f1_rf = 0


print("\n========================================")
print("MÉTRICAS - RANDOM FOREST")
print("========================================")

print("VP:", VP)
print("VN:", VN)
print("FP:", FP)
print("FN:", FN)

print(f"\nAccuracy: {accuracy_rf:.3f}")
print(f"Precision: {precision_rf:.3f}")
print(f"Recall: {recall_rf:.3f}")
print(f"F1-score: {f1_rf:.3f}")

# ============================================================
#  PUNTO 4 - PREDICCIONES DE CADA ÁRBOL
# ============================================================

def predicciones_individuales(bosque, X):
    resultados = []

    for arbol in bosque:
        predicciones = []

        for _, fila in X.iterrows():
            prediccion = predecir_arbol(arbol, fila)
            predicciones.append(prediccion)

        resultados.append(predicciones)

    return resultados


# Obtener las predicciones de los 10 árboles
predicciones_arboles = predicciones_individuales(bosque, X_test)

print("\n============================================================")
print("PREDICCIONES INDIVIDUALES DE LOS 10 ÁRBOLES")
print("============================================================")

# Mostrar las primeras 20 personas del conjunto de prueba
for i in range(min(20, len(X_test))):

    votos = []

    for numero_arbol in range(len(predicciones_arboles)):
        prediccion = predicciones_arboles[numero_arbol][i]
        votos.append(prediccion)

    cantidad_0 = votos.count(0)
    cantidad_1 = votos.count(1)

    if cantidad_1 > cantidad_0:
        prediccion_final = 1
    else:
        prediccion_final = 0

    print(
        f"Registro {i + 1} | "
        f"Real: {y_test.iloc[i]} | "
        f"Árboles: {votos} | "
        f"Voto final: {prediccion_final}"
    )
