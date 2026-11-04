import sys
import numpy as np
import pandas as pd

SEED = 42
RUTA = sys.argv[1] if len(sys.argv) > 1 else "TP N°3/student_performance_dataset.csv"

# ----------------------------------------------------------------------
# Lectura y preparacion de datos
# ----------------------------------------------------------------------
df = pd.read_csv(RUTA)
print("Filas originales:", len(df))
df = df.dropna()                                      # hay filas con valores faltantes
df = df.drop_duplicates().reset_index(drop=True)      # hay filas repetidas en el CSV
print("Filas sin duplicados:", len(df))

COLUMNAS = ["Study_Hours_per_Week", "Past_Exam_Scores", "Attendance_Rate"]
NOMBRES = ["Horas de estudio", "Notas previas", "Asistencia"]
X = df[COLUMNAS].to_numpy(dtype=float)                # orden: horas, previas, asistencia
y_nota = df["Final_Exam_Score"].to_numpy(dtype=float)
y = (df["Pass_Fail"] == "Pass").to_numpy().astype(int)  # 1 = aprobado, 0 = reprobado


# ----------------------------------------------------------------------
# Ejercicio 1.1: particion 80/20 aleatoria (estratificada)
# ----------------------------------------------------------------------
def split_estratificado(y, prop_test=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    test = []
    for clase in (0, 1):
        ids = rng.permutation(np.where(y == clase)[0])
        test.extend(ids[: int(round(prop_test * len(ids)))])
    test = np.array(sorted(test))
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test


tr, te = split_estratificado(y)
Xtr, Xte = X[tr], X[te]
ytr, yte = y[tr], y[te]
ntr, nte = y_nota[tr], y_nota[te]
print(f"Entrenamiento: {len(tr)}  Prueba: {len(te)}")

# estandarizacion con estadisticos del conjunto de entrenamiento
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd


# ----------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------
def matriz_confusion(real, pred):
    tn = int(np.sum((real == 0) & (pred == 0)))
    fp = int(np.sum((real == 0) & (pred == 1)))
    fn = int(np.sum((real == 1) & (pred == 0)))
    tp = int(np.sum((real == 1) & (pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def metricas(real, pred):
    (tn, fp), (fn, tp) = matriz_confusion(real, pred)
    acc = (tp + tn) / len(real)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, f1


def r2(real, pred):
    return 1 - np.sum((real - pred) ** 2) / np.sum((real - real.mean()) ** 2)


# ----------------------------------------------------------------------
# Ejercicio 1.2: regresion lineal multiple (minimos cuadrados)
# ----------------------------------------------------------------------
A_tr = np.column_stack([np.ones(len(Xtr)), Xtr])
A_te = np.column_stack([np.ones(len(Xte)), Xte])
beta_lin = np.linalg.lstsq(A_tr, ntr, rcond=None)[0]

print("\n=== Ejercicio 1.2: Regresion lineal multiple ===")
print(f"Nota = {beta_lin[0]:.4f} + {beta_lin[1]:.4f}*Horas + "
      f"{beta_lin[2]:.4f}*NotasPrevias + {beta_lin[3]:.4f}*Asistencia")
print(f"R2 entrenamiento: {r2(ntr, A_tr @ beta_lin):.4f}")
print(f"R2 prueba:        {r2(nte, A_te @ beta_lin):.4f}")


# ----------------------------------------------------------------------
# Ejercicio 1.3: regresion logistica (maximo verosimilitud, Newton-Raphson)
# ----------------------------------------------------------------------
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def ajustar_logistica(Z, y, iteraciones=100, tol=1e-10):
    A = np.column_stack([np.ones(len(Z)), Z])
    b = np.zeros(A.shape[1])
    for _ in range(iteraciones):
        p = sigmoide(A @ b)
        W = p * (1 - p)
        grad = A.T @ (y - p)
        H = A.T @ (A * W[:, None])
        paso = np.linalg.solve(H + 1e-10 * np.eye(len(b)), grad)
        b = b + paso
        if np.max(np.abs(paso)) < tol:
            break
    return b


b_std = ajustar_logistica(Ztr, ytr)
# volver a las unidades originales de las variables
beta_log = np.empty(4)
beta_log[1:] = b_std[1:] / sd
beta_log[0] = b_std[0] - np.sum(b_std[1:] * mu / sd)


def prob_logistica(Xnuevo):
    return sigmoide(np.column_stack([np.ones(len(Xnuevo)), Xnuevo]) @ beta_log)


print("\n=== Ejercicio 1.3: Regresion logistica ===")
print(f"ln(p/(1-p)) = {beta_log[0]:.4f} + {beta_log[1]:.4f}*Horas + "
      f"{beta_log[2]:.4f}*NotasPrevias + {beta_log[3]:.4f}*Asistencia")
pred_log = (prob_logistica(Xte) >= 0.5).astype(int)
cm = matriz_confusion(yte, pred_log)
acc, f1 = metricas(yte, pred_log)
print("Matriz de confusion (filas: real 0/1, columnas: predicho 0/1):")
print(cm)
print(f"Accuracy: {acc:.4f}   F1-score: {f1:.4f}")

# ----------------------------------------------------------------------
# Ejercicio 1.4: prediccion para el estudiante pedido
# ----------------------------------------------------------------------
nuevo = np.array([[25.0, 68.0, 58.0]])      # horas, notas previas, asistencia
nota_pred = float(np.array([1.0, *nuevo[0]]) @ beta_lin)
p_nuevo = float(prob_logistica(nuevo)[0])
print("\n=== Ejercicio 1.4: estudiante (25 h, 58 % asistencia, 68 previas) ===")
print(f"Regresion lineal:   nota predicha = {nota_pred:.2f} -> "
      f"{'APROBADO' if nota_pred >= 60 else 'REPROBADO'} (umbral 60)")
print(f"Regresion logistica: P(aprobar) = {p_nuevo:.4f} -> "
      f"{'APROBADO' if p_nuevo >= 0.5 else 'REPROBADO'}")



#--------------------------------------------------------------------------------------------


#Esto está de más, pero lo dejo para que no se pierda el contexto

# Ejercicio 2: SVM (formulacion dual, resuelta con SMO), solo numpy

GAMMA = 1.0 / Ztr.shape[1]                  # equivale a 'scale' con datos estandarizados


def kernel(nombre, A, B):
    if nombre == "lineal":
        return A @ B.T
    if nombre == "polinomico":              # (gamma <x,z> + 1)^3
        return (GAMMA * (A @ B.T) + 1.0) ** 3
    if nombre == "rbf":                     # exp(-gamma ||x-z||^2)
        d2 = (np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T)
        return np.exp(-GAMMA * d2)
    raise ValueError(nombre)


class SVM:
    def __init__(self, kern="lineal", C=1.0, tol=1e-3, max_pasadas=20, seed=SEED):
        self.kern, self.C, self.tol, self.max_pasadas = kern, C, tol, max_pasadas
        self.rng = np.random.default_rng(seed)

    def fit(self, Z, y01):
        y = 2.0 * y01 - 1.0                       # etiquetas en {-1, +1}
        n = len(y)
        K = kernel(self.kern, Z, Z)
        alpha, b = np.zeros(n), 0.0
        pasadas, iteraciones = 0, 0
        while pasadas < self.max_pasadas:
            cambios = 0
            for i in range(n):
                Ei = (alpha * y) @ K[:, i] + b - y[i]
                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and alpha[i] > 0):
                    j = self.rng.integers(n - 1)
                    j = j + 1 if j >= i else j     # j != i
                    Ej = (alpha * y) @ K[:, j] + b - y[j]
                    ai, aj = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L, H = max(0, aj - ai), min(self.C, self.C + aj - ai)
                    else:
                        L, H = max(0, ai + aj - self.C), min(self.C, ai + aj)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    alpha[j] = np.clip(aj - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(alpha[j] - aj) < 1e-5:
                        continue
                    alpha[i] = ai + y[i] * y[j] * (aj - alpha[j])
                    b1 = b - Ei - y[i] * (alpha[i] - ai) * K[i, i] - y[j] * (alpha[j] - aj) * K[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - ai) * K[i, j] - y[j] * (alpha[j] - aj) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    cambios += 1
            pasadas = pasadas + 1 if cambios == 0 else 0
            iteraciones += 1
            if iteraciones > 400:                  # tope de seguridad
                break
        self.Z, self.ay, self.b = Z, alpha * y, b
        self.n_sv = int(np.sum(alpha > 1e-8))
        return self

    def decision(self, Zn):
        return kernel(self.kern, Zn, self.Z) @ self.ay + self.b

    def predict(self, Zn):
        return (self.decision(Zn) >= 0).astype(int)


print("\n=== Ejercicio 2: SVM ===")
print(f"{'Kernel':<11}{'C':>6}{'TN':>5}{'FP':>5}{'FN':>5}{'TP':>5}{'Accuracy':>10}{'F1':>8}{'Nuevo':>10}")
Znuevo = (nuevo - mu) / sd
resultados = []
for kern in ("lineal", "polinomico", "rbf"):
    for C in (0.1, 1, 10, 100):
        modelo = SVM(kern, C).fit(Ztr, ytr)
        pred = modelo.predict(Zte)
        (tn, fp), (fn, tp) = matriz_confusion(yte, pred)
        acc, f1 = metricas(yte, pred)
        clase = "APROBADO" if modelo.predict(Znuevo)[0] == 1 else "REPROBADO"
        resultados.append((kern, C, tn, fp, fn, tp, acc, f1, clase))
        print(f"{kern:<11}{C:>6}{tn:>5}{fp:>5}{fn:>5}{tp:>5}{acc:>10.4f}{f1:>8.4f}{clase:>10}")

# ----------------------------------------------------------------------
# Ejercicio 3: comparacion
# ----------------------------------------------------------------------
print("\n=== Ejercicio 3: comparacion ===")
print(f"Regresion logistica -> Accuracy {metricas(yte, pred_log)[0]:.4f}, F1 {metricas(yte, pred_log)[1]:.4f}")
mejor = max(resultados, key=lambda r: (r[7], r[6]))
print(f"Mejor SVM (por F1)   -> {mejor[0]}, C={mejor[1]}: Accuracy {mejor[6]:.4f}, F1 {mejor[7]:.4f}")
print(f"Referencia: predecir siempre la clase mayoritaria -> accuracy {max(yte.mean(), 1 - yte.mean()):.4f}")
