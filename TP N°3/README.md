# Aprendizaje Automático — Trabajo Práctico 3

Trabajo práctico desarrollado en **Python** para aplicar técnicas de **Aprendizaje Automático** sobre un conjunto de datos de estudiantes y analizar su rendimiento en un examen.

## Objetivo

Predecir el rendimiento de un estudiante utilizando como variables principales:

- Horas de estudio por semana.
- Calificaciones obtenidas previamente.
- Porcentaje de asistencia.

## Trabajo realizado

Se dividió el dataset en un **80% para entrenamiento y 20% para prueba**.

Se implementaron y compararon diferentes modelos:

- **Regresión Lineal Múltiple**, para predecir la calificación obtenida en el examen.
- **Regresión Logística**, para determinar si un estudiante aprueba o no.
- **SVM**, probando diferentes kernels y valores del parámetro `C`.

Para evaluar los modelos de clasificación se utilizaron **matrices de confusión, Accuracy y F1-Score**, mientras que para la regresión lineal se utilizó el **coeficiente de determinación R²**.

También se realizaron predicciones para un estudiante con **25 horas de estudio semanales, 58% de asistencia y 68 puntos en exámenes previos**.

## Aprendizajes

Durante el trabajo se aprendió a preparar y dividir datos, entrenar modelos de regresión y clasificación, evaluar sus resultados y comparar diferentes métodos para determinar cuál se adapta mejor a un problema de predicción.

Finalmente, se analizó cuál de los modelos implementados resulta más adecuado para predecir la **condición de aprobación de un estudiante**.
