import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# QUESTÃO 1 e 4: Estimadores de MQO e Máxima Verossimilhança (Conceitual / Teórico)
print("\n[Questão 1 & 4] As demonstrações analíticas dos Estimadores de Mínimos Quadrados")
print( "e de Máxima Verossimilhança para beta_0, beta_1 e sigma^2 foram detalhadas nas respostas anteriores")

# QUESTÃO 7: Distribuição Conjunta, Esperança, Variância e Correlação
# Matriz de probabilidades conjuntas P(X=x_i, Y=y_j)
P = np.array([[0.2, 0.1, 0.0], [0.1, 0.2, 0.1], [0.0, 0.1, 0.2]])

x_vals = np.array([2, 4, 6])
y_vals = np.array([4, 5, 6])

p_x = P.sum(axis=0)  # de X
p_y = P.sum(axis=1)  # de Y

e_x = np.sum(x_vals * p_x)
e_y = np.sum(y_vals * p_y)

e_x2 = np.sum((x_vals ** 2) * p_x)
e_y2 = np.sum((y_vals ** 2) * p_y)

var_x = e_x2 - e_x ** 2
var_y = e_y2 - e_y ** 2

e_xy = 0
for i, y in enumerate(y_vals):
  for j, x in enumerate(x_vals):
    e_xy += x * y * P[i, j]

cov_xy = e_xy - (e_x * e_y)
corr_rho = cov_xy / np.sqrt(var_x * var_y)

print(f"a) As variáveis são independentes? Não, pois P(X=2, Y=4) = 0.2 != P(X=2)*P(Y=4) = {p_x[0]*p_y[0]:.4f}")
print(f"b) E[X] = {e_x:.4f} | E[Y] = {e_y:.4f}")
print(f"c) Var(X) = {var_x:.4f} | Var(Y) = {var_y:.4f}")
print(f"d) Cov(X, Y) = {cov_xy:.4f} | Correlação (rho) = {corr_rho:.4f}")

# QUESTÃO 8: Ciclistas e Chumbo no Sangue
horas = np.array([8, 10, 10, 12, 15, 18, 18, 21, 25, 25], dtype=float)
chumbo = np.array([0.53, 0.25, 0.34, 0.25, 0.29, 0.30, 0.53, 0.53, 0.53, 0.87])

X_8 = sm.add_constant(horas)
mod_8_10 = sm.OLS(chumbo, X_8).fit()
mod_8_9 = sm.OLS(chumbo[:-1], sm.add_constant(horas[:-1])).fit()

print("Modelo com 10 ciclistas")
print(f"Equação: Y = {mod_8_10.params[0]:.4f} + {mod_8_10.params[1]:.4f}X")
print(f"R² = {mod_8_10.rsquared:.4f}")

print("\nModelo sem o ciclista 10 (n=9)")
print(f"Equação: Y = {mod_8_9.params[0]:.4f} + {mod_8_9.params[1]:.4f}X")
print(f"R² = {mod_8_9.rsquared:.4f}")

# QUESTÃO 9: Pesos de Pais e Filhos
X_pais = np.array([60, 65, 70, 68, 63, 69, 71, 64, 66, 64], dtype=float)
Y_filhos = np.array([63, 64, 71, 69, 63, 68, 73, 63, 64, 62], dtype=float)

Z = X_pais - 66
W = Y_filhos - 67

r_xy = np.corrcoef(X_pais, Y_filhos)[0, 1]
r_zw = np.corrcoef(Z, W)[0, 1]

s_x = np.std(X_pais, ddof=1)
s_z = np.std(Z, ddof=1)
s_y = np.std(Y_filhos, ddof=1)
s_w = np.std(W, ddof=1)

b_orig = np.cov(X_pais, Y_filhos)[0, 1] / np.var(X_pais, ddof=1)
b_trans = np.cov(Z, W)[0, 1] / np.var(Z, ddof=1)

print(f"a) Correlação r_XY = {r_xy:.4f} | Correlação r_ZW = {r_zw:.4f}")
print(f"b) Sx = {s_x:.4f}, Sz = {s_z:.4f} | Sy = {s_y:.4f}, Sw = {s_w:.4f}")
print(f"c) Coeficiente angular b (original) = {b_orig:.4f} | (trans) = {b_trans:.4f}")
print(
    f"d) R² (original) = {r_xy**2:.4f} | R² (transformado) = {r_zw**2:.4f}"
)

df_9 = pd.DataFrame({"X": X_pais, "Y": Y_filhos})
mod_9 = smf.ols("Y ~ X", data=df_9).fit()
anova_9 = sm.stats.anova_lm(mod_9)
print("\ne) Quadro da ANOVA (Modelo 9):")
print(anova_9)

# QUESTÃO 10: Massa Muscular vs Idade de 18 Mulheres
massa_y = np.array([
    82.0, 91.0, 100.0, 68.0, 87.0, 73.0, 78.0, 80.0, 65.0,
    84.0, 116.0, 76.0, 97.0, 100.0, 105.0, 77.0, 73.0, 78.0
])
idade_x = np.array([
    71.0, 64.0, 43.0, 67.0, 56.0, 73.0, 68.0, 56.0, 76.0,
    65.0, 45.0, 58.0, 45.0, 53.0, 49.0, 78.0, 73.0, 68.0
])

df_10 = pd.DataFrame({"X": idade_x, "Y": massa_y})
mod_10 = smf.ols("Y ~ X", data=df_10).fit()
r_10 = np.corrcoef(idade_x, massa_y)[0, 1]

print(f"b) Coeficiente de correlação linear (r) = {r_10:.4f}")
print(f"c) Parâmetros estimados: Beta_0 = {mod_10.params['Intercept']:.4f}, Beta_1 = {mod_10.params['X']:.4f}")
print(f"d) Coeficiente de determinação (R²) = {mod_10.rsquared:.4f}")
print("\ne) Quadro da ANOVA (Modelo 10):")
print(sm.stats.anova_lm(mod_10))

# QUESTÃO 11: Ruibarbo (Blocos ao Acaso e Falta de Ajuste)
dados_11 = np.array([
    [21.2, 21.4, 12.0, 17.2],
    [19.3, 17.4, 24.5, 30.2],
    [22.8, 29.0, 18.5, 24.5],
    [26.0, 34.0, 33.0, 30.2],
    [43.5, 37.0, 21.5, 23.5],
    [32.1, 30.5, 35.7, 32.3],
    [33.0, 32.2, 35.4, 35.4]
])

n_trat = 7
n_blocos = 4

tratamentos = np.repeat(np.arange(1, n_trat + 1), n_blocos)
blocos = np.tile(np.arange(1, n_blocos + 1), n_trat)
y_11 = dados_11.flatten()

df_11 = pd.DataFrame({
    "Y": y_11,
    "Tratamento": pd.Categorical(tratamentos),
    "Bloco": pd.Categorical(blocos)
})

mod_11_rcb = smf.ols("Y ~ C(Tratamento) + C(Bloco)", data=df_11).fit()
print("a) ANOVA - Delineamento em Blocos ao Acaso:")
print(sm.stats.anova_lm(mod_11_rcb))

medias_trat = dados_11.mean(axis=1)
x_num = np.arange(1, n_trat + 1, dtype=float)
mod_reg_linear = smf.ols(
    "Y ~ X", data=pd.DataFrame({"X": x_num, "Y": medias_trat})
).fit()

print(f"\nb) Regressão Linear sobre Tratamentos: R² = {mod_reg_linear.rsquared:.4f}")
print("c) Equação da reta para as médias de tratamentos:")
print(f"   Média_Y = {mod_reg_linear.params['Intercept']:.4f} + {mod_reg_linear.params['X']:.4f} * Tratamento")

# QUESTÃO 12: Temperatura vs Tempo de Reação Química
temperaturas = np.array([20, 30, 40, 50, 60, 70, 80])
tempos_matriz = np.array([
    [12.3, 11.8, 11.5, 12.1, 11.7],
    [11.8, 11.5, 11.4, 11.7, 11.2],
    [10.9, 11.2, 10.8, 10.6, 10.3],
    [10.4, 9.8, 9.5, 9.9, 9.2],
    [9.6, 9.0, 8.7, 8.3, 9.1],
    [9.1, 9.3, 8.5, 8.6, 8.3],
    [8.4, 8.1, 8.1, 7.7, 7.9]
])

X_12 = np.repeat(temperaturas, 5)
Y_12 = tempos_matriz.flatten()

df_12 = pd.DataFrame({"X": X_12, "Y": Y_12, "X_cat": pd.Categorical(X_12)})
mod_12 = smf.ols("Y ~ X", data=df_12).fit()

print(f"b) Média global = {Y_12.mean():.4f} | Variância global = {Y_12.var(ddof=1):.4f}")
print("c) Média e variância por temperatura fixada:")
for i, temp in enumerate(temperaturas):
  sub = tempos_matriz[i]
  print(
      f"   Temp {temp}°C -> Média: {sub.mean():.4f}, Variância: {sub.var(ddof=1):.4f}"
  )

print(f"\nd) Correlação linear (r) = {np.corrcoef(X_12, Y_12)[0,1]:.4f}")
print(f"e) Estimativas dos parâmetros: Beta_0 = {mod_12.params['Intercept']:.4f}, Beta_1 = {mod_12.params['X']:.4f}")
print(f"f) Coeficiente de determinação (R²) = {mod_12.rsquared:.4f}")

mod_linear = smf.ols("Y ~ X", data=df_12).fit()
mod_fator = smf.ols("Y ~ X_cat", data=df_12).fit()