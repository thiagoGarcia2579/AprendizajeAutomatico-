import pandas as pd

# Cargar los datos
datos = pd.read_csv("Prestamo.csv", encoding="latin1")

datos_originales = datos.copy()

if "Edad" in datos.columns:
    datos = datos[datos["Edad"] == 50].reset_index(drop=True)

# 1. DIVISIÓN DEL CONJUNTO (75% Entrenamiento / 25% Prueba)
n_entrenamiento = int(len(datos) * 0.75)

entrenamiento = datos.iloc[:n_entrenamiento].copy()
testeo = datos.iloc[n_entrenamiento:].copy()

print(f"Total de ejemplos de 50 años: {len(datos)}")
print(f"Entrenamiento: {len(entrenamiento)} | Prueba: {len(testeo)}\n")

# Atributos que vamos a utilizar
atributos = [
    "Sexo",
    "Mayor nivel educativo",
    "Estado de vivienda",
    "Préstamos previos impagos"
]

# Función FIND-S
def find_s(datos_entrenamiento):
    # La hipótesis comienza siendo totalmente específica
    hipotesis = None

    for _, ejemplo in datos_entrenamiento.iterrows():

        if ejemplo["Estado"] == "OTORGADO":
            valores = [ejemplo[atributo] for atributo in atributos]

            # Primer ejemplo positivo
            hipotesis = inicializar_hipotesis(hipotesis, valores)
    return hipotesis

def inicializar_hipotesis(hipotesis, valores):
    if hipotesis is None:
        hipotesis = valores
    else:
        comparar_con_hipotesis_actual(hipotesis, valores)
    return hipotesis

def comparar_con_hipotesis_actual(hipotesis, valores):
            # Comparar con la hipótesis actual
            for i in range(len(atributos)):
                if hipotesis[i] != valores[i]:
                    hipotesis[i] = "?"

# Obtener la hipótesis final y mostrarla
def obtener_hipotesis_final():
    hipotesis = find_s(entrenamiento)
    print("Hipótesis final obtenida por FIND-S:")
    for atributo, valor in zip(atributos, hipotesis):
        print(f"{atributo}: {valor}")

obtener_hipotesis_final()

# 3. Aplicar la hipótesis al conjunto de prueba

def predecir(ejemplo, hipotesis):

    for i, atributo in enumerate(atributos):

        if hipotesis[i] != "?" and ejemplo[atributo] != hipotesis[i]:
            return "RECHAZADO"

    return "OTORGADO"


def realizar_predicciones():
    hipotesis = find_s(entrenamiento)

    testeo["Prediccion"] = testeo.apply(
        lambda fila: predecir(fila, hipotesis),
        axis=1
    )

    print("\nPredicciones:")
    print(testeo[atributos + ["Estado", "Prediccion"]])


realizar_predicciones()

# Ejercicio 2

def matriz_confusion(y_real, y_pred):

    VP = 0
    VN = 0
    FP = 0
    FN = 0

    for real, prediccion in zip(y_real, y_pred):

        if real == "OTORGADO" and prediccion == "OTORGADO":
            VP += 1

        elif real == "RECHAZADO" and prediccion == "RECHAZADO":
            VN += 1

        elif real == "RECHAZADO" and prediccion == "OTORGADO":
            FP += 1

        elif real == "OTORGADO" and prediccion == "RECHAZADO":
            FN += 1

    return VP, VN, FP, FN

VP, VN, FP, FN = matriz_confusion(y_real=testeo["Estado"], y_pred=testeo["Prediccion"])
print(f"\nMatriz de Confusión:")
print(f"VP: {VP}, VN: {VN}, FP: {FP}, FN: {FN}")

def calcular_metricas(VP, VN, FP, FN):

    accuracy = (VP + VN) / (VP + VN + FP + FN)

    recall = VP / (VP + FN)

    especificidad = VN / (VN + FP)

    precision = VP / (VP + FP)

    print("\nMétricas:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"Precisión: {precision:.4f}")

    return accuracy, recall, especificidad, precision


accuracy, recall, especificidad, precision = calcular_metricas(
    VP, VN, FP, FN
)

def calcular_f1(precision, recall):

    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"\nF1-score: {f1:.4f}")

    return f1


f1 = calcular_f1(precision, recall)


def calcular_tasas(VP, VN, FP, FN):

    TPR = VP / (VP + FN)

    FPR = FP / (FP + VN)

    print(f"\nTasa de verdaderos positivos (TPR): {TPR:.4f}")
    print(f"Tasa de falsos positivos (FPR): {FPR:.4f}")

    return TPR, FPR


TPR, FPR = calcular_tasas(VP, VN, FP, FN)

#Ejercicio 3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

datos_40_45 = datos_originales[
    (datos_originales["Edad"] >= 40) &
    (datos_originales["Edad"] <= 45)
].copy()

# Separar atributos y resultado
X = datos_40_45[atributos]
y = datos_40_45["Estado"]


# División aleatoria: 80% entrenamiento / 20% prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nEjercicio 3")
print(f"Total de ejemplos entre 40 y 45 años: {len(datos_40_45)}")
print(f"Entrenamiento: {len(X_entrenamiento)}")
print(f"Prueba: {len(X_prueba)}")


# 2. CLASIFICADOR NAÏVE BAYES

# Convertir los atributos categóricos a números
encoder = OrdinalEncoder()

X_entrenamiento_codificado = encoder.fit_transform(X_entrenamiento)
X_prueba_codificada = encoder.transform(X_prueba)


# Crear y entrenar el modelo
modelo = CategoricalNB()

modelo.fit(
    X_entrenamiento_codificado,
    y_entrenamiento
)


# Clasificar conjunto de prueba
y_pred = modelo.predict(X_prueba_codificada)


print("\nPredicciones:")
print(y_pred)


# =========================
# 3. MATRIZ DE CONFUSIÓN
# =========================

VP, VN, FP, FN = matriz_confusion(y_prueba, y_pred)

print("\nMatriz de confusión:")
print(f"Verdaderos positivos: {VP}")
print(f"Verdaderos negativos: {VN}")
print(f"Falsos positivos: {FP}")
print(f"Falsos negativos: {FN}")


# =========================
# 4. ACCURACY Y F1-SCORE
# =========================

accuracy = (VP + VN) / (VP + VN + FP + FN)

recall = VP / (VP + FN)

precision = VP / (VP + FP)

f1 = 2 * (precision * recall) / (precision + recall)


print("\nMétricas:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")


# =========================
# 5. CURVA ROC
# =========================

# Convertir OTORGADO/RECHAZADO a valores 1/0
y_prueba_binario = (y_prueba == "OTORGADO").astype(int)

# Obtener probabilidad de OTORGADO
probabilidades = modelo.predict_proba(X_prueba_codificada)

# Buscar qué columna corresponde a OTORGADO
indice_otorgado = list(modelo.classes_).index("OTORGADO")

probabilidad_otorgado = probabilidades[:, indice_otorgado]


# Calcular TPR y FPR
FPR, TPR, umbrales = roc_curve(
    y_prueba_binario,
    probabilidad_otorgado
)


# Calcular AUC
valor_auc = auc(FPR, TPR)


# Graficar
plt.figure()

plt.plot(FPR, TPR, label=f"AUC = {valor_auc:.2f}")

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("Tasa de falsos positivos (FPR)")
plt.ylabel("Tasa de verdaderos positivos (TPR)")
plt.title("Curva ROC - Naïve Bayes")

plt.legend()
plt.grid()

plt.show()