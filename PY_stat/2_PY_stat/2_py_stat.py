import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

# Carregando os dados
pasta_atual = Path(__file__).resolve().parent

df = pd.read_csv(pasta_atual / "dados_estudo - dados_estudo.csv")
print("Conhecendo a base")
print(df.info())
print(df[['horas_estudo', 'nota']].describe())

# Visualização
plt.scatter(df['horas_estudo'], df['nota'], color='blue', label='Observações')
plt.title('Nota vs Horas de Estudo')
plt.xlabel('Horas Semanais de Estudo')
plt.ylabel('Nota')

# Correlação
print("\nCorrelação")
print("Correlação de Pearson:", df['horas_estudo'].corr(df['nota']))

# Ajuste do modelo
X = df['horas_estudo']
Y = df['nota']
X_sm = sm.add_constant(X) 
modelo = sm.OLS(Y, X_sm).fit()

# Adicionar reta
plt.plot(X, modelo.predict(X_sm), color='red', label='Reta Ajustada')
plt.legend()
plt.show()

print("\nResumo do Modelo OLS")
print(modelo.summary())

# Valores ajustados e resíduos
df['y_ajustado'] = modelo.predict(X_sm)
df['residuo'] = modelo.resid
print("\nPrimeiros 5 registros (X, Y, Y_ajustado, Residuo)")
print(df[['horas_estudo', 'nota', 'y_ajustado', 'residuo']].head())

print("\nSoma dos residuos:", np.sum(modelo.resid))
print("Soma X * residuos:", np.sum(X * modelo.resid))

# Tabela ANOVA
print("\nANOVA")
print("F-statistic:", modelo.fvalue)
print("SQReg:", np.sum((df['y_ajustado'] - Y.mean())**2))
print("SQRes:", np.sum(modelo.resid**2))

# Intervalos de confiança
print("\nIntervalos de Confiança (95%)")
print(modelo.conf_int(alpha=0.05))

# Predições
x_novos = pd.DataFrame({'const': [1, 1], 'horas_estudo': [6, 10]})
previsoes = modelo.get_prediction(x_novos).summary_frame(alpha=0.05)
print("\nPrevisões (X=6 e X=10)")
print(previsoes[['mean', 'mean_ci_lower', 'mean_ci_upper', 'obs_ci_lower', 'obs_ci_upper']])

# Diagnóstico do Modelo Gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Residuos vs Ajustados
ax1.scatter(df['y_ajustado'], df['residuo'])
ax1.axhline(0, color='red', linestyle='--')
ax1.set_title('Resíduos vs Ajustados')
ax1.set_xlabel('Valores Ajustados')
ax1.set_ylabel('Resíduos')

# Q-Q Plot
sm.qqplot(df['residuo'], line='45', fit=True, ax=ax2)
ax2.set_title('Gráfico Q-Q dos Resíduos')
plt.show()