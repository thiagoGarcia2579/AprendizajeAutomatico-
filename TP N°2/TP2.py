import pandas as pd
import random
import math


# ============================================================
# 1. CARGAR LOS DATOS
# ============================================================

df = pd.read_csv("loan_data.csv")

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

# Variable que queremos predecir
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
# 6. ENTROPÍA DE SHANNON
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
# 7. GANANCIA DE INFORMACIÓN
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
# 8. MOSTRAR GANANCIA DE CADA ATRIBUTO
# ============================================================

print("\nGanancia de información:")

for atributo in atributos:

    ganancia = ganancia_informacion(
        X_train,
        y_train,
        atributo
    )

    print(
        atributo,
        "->",
        ganancia
    )


# ============================================================
# 9. ALGORITMO ID3
# ============================================================

def id3(X, y, atributos):

    # --------------------------------------------------------
    # CASO 1:
    # Todos los ejemplos pertenecen a la misma clase
    # --------------------------------------------------------

    if len(y.unique()) == 1:

        return int(y.iloc[0])


    # --------------------------------------------------------
    # CASO 2:
    # No quedan atributos
    # --------------------------------------------------------

    if len(atributos) == 0:

        return int(y.mode()[0])


    # --------------------------------------------------------
    # BUSCAR EL ATRIBUTO CON MAYOR GANANCIA
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SI NO HAY GANANCIA, DEVOLVER CLASE MAYORITARIA
    # --------------------------------------------------------

    if mayor_ganancia <= 0:

        return int(y.mode()[0])


    # --------------------------------------------------------
    # CREAR NODO
    # --------------------------------------------------------

    arbol = {
        mejor_atributo: {}
    }


    # --------------------------------------------------------
    # ELIMINAR ATRIBUTO UTILIZADO
    # --------------------------------------------------------

    atributos_restantes = [
        atributo
        for atributo in atributos
        if atributo != mejor_atributo
    ]


    # --------------------------------------------------------
    # CREAR RAMAS
    # --------------------------------------------------------

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
# 10. CONSTRUIR EL ÁRBOL
# ============================================================

arbol = id3(
    X_train,
    y_train,
    atributos
)


# ============================================================
# 11. MOSTRAR EL ÁRBOL DE FORMA LEGIBLE
# ============================================================

def imprimir_arbol(arbol, nivel=0):

    espacios = "    " * nivel

    # Si encontramos una hoja (0 o 1)
    if not isinstance(arbol, dict):

        print(espacios + "→ " + str(arbol))
        return


    # Obtener el atributo del nodo
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
# 12. FUNCIÓN PARA PREDECIR CON EL ÁRBOL ID3
# ============================================================

def predecir(arbol, ejemplo):

    # Si llegamos a una hoja
    if not isinstance(arbol, dict):
        return arbol

    # Obtener el atributo utilizado en este nodo
    atributo = next(iter(arbol))

    # Obtener el valor que tiene el ejemplo
    valor = ejemplo[atributo]

    # Si existe una rama para ese valor
    if valor in arbol[atributo]:

        return predecir(
            arbol[atributo][valor],
            ejemplo
        )

    # Si aparece un valor que no estaba en entrenamiento,
    # devolvemos la clase más frecuente
    return 0


# ============================================================
# 13. REALIZAR PREDICCIONES SOBRE LOS DATOS DE PRUEBA
# ============================================================

predicciones = []

for indice in X_test.index:

    ejemplo = X_test.loc[indice]

    prediccion = predecir(
        arbol,
        ejemplo
    )

    predicciones.append(prediccion)


print("\n========================================")
print("PREDICCIONES")
print("========================================")

print("Cantidad de predicciones:", len(predicciones))

print("\nPrimeras predicciones:")
print(predicciones[:20])


# ============================================================
# 14. MATRIZ DE CONFUSIÓN
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
print("MATRIZ DE CONFUSIÓN")
print("========================================")

print("                 Predicho")
print("              0          1")
print()
print("Real  0      ", VN, "       ", FP)
print("      1      ", FN, "       ", VP)


# ============================================================
# 15. MÉTRICAS
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
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0


# ============================================================
# 16. MOSTRAR RESULTADOS
# ============================================================

print("\n========================================")
print("MÉTRICAS")
print("========================================")

print("VP:", VP)
print("VN:", VN)
print("FP:", FP)
print("FN:", FN)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)