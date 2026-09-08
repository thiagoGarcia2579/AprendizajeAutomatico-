import pandas as pd

# Cargar los datos
datos = pd.read_csv("Prestamo.csv", encoding="latin1")

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


    
