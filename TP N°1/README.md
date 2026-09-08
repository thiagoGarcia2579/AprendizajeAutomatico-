# TP N°1 — Aprendizaje Automático

Trabajo Práctico N°1 de la materia **Aprendizaje Automático**.

### Contenidos

* FIND-S
* Naïve Bayes
* Matriz de confusión
* Métricas de evaluación
* Curva ROC

### Matriz de Confusión

* **VP (Verdadero Positivo):** casos que eran `OTORGADO` y fueron clasificados como `OTORGADO`.
* **VN (Verdadero Negativo):** casos que eran `RECHAZADO` y fueron clasificados como `RECHAZADO`.
* **FP (Falso Positivo):** casos que eran `RECHAZADO` pero fueron clasificados como `OTORGADO`.
* **FN (Falso Negativo):** casos que eran `OTORGADO` pero fueron clasificados como `RECHAZADO`.

### Métricas

* **Accuracy:** proporción de predicciones correctas sobre el total.
  **(VP + VN) / (VP + VN + FP + FN)**

* **Recall:** proporción de casos `OTORGADO` correctamente identificados.
  **VP / (VP + FN)**

* **Especificidad:** proporción de casos `RECHAZADO` correctamente identificados.
  **VN / (VN + FP)**

* **Precisión:** proporción de predicciones `OTORGADO` que fueron correctas.
  **VP / (VP + FP)**

* **F1-score:** combina Precisión y Recall mediante su media armónica.
  **2 × (Precisión × Recall) / (Precisión + Recall)**

* **TPR:** tasa de verdaderos positivos. Es equivalente al Recall.
  **VP / (VP + FN)**

* **FPR:** tasa de falsos positivos. Indica qué proporción de los `RECHAZADO` fue clasificada incorrectamente como `OTORGADO`.
  **FP / (FP + VN)**

### Tecnologías

* Python
* Pandas
* Scikit-learn
* Matplotlib

