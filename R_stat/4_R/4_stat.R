setwd("/home/maicongomes/Downloads/stat-modeling/R_stat/4_R")

# Conhecendo a base
dados <- read.csv("dados_estudo - dados_estudo.csv")
cat("Informações da Base\n")
str(dados)
summary(dados)
cat("Desvios Padrão:\nX:", sd(dados$horas_estudo), "Y:", sd(dados$nota), "\n")

# Correlação
cat("\nCorrelação\n")
cat("Correlação de Pearson:", cor(dados$horas_estudo, dados$nota), "\n")

# Ajuste do modelo
modelo <- lm(nota ~ horas_estudo, data = dados)
cat("\n== Sumário do Modelo ==\n")
summary(modelo)

# Visualização
plot(dados$horas_estudo, dados$nota, 
     main = "Nota vs Horas de Estudo", 
     xlab = "Horas Semanais de Estudo", 
     ylab = "Nota", pch = 16, col = "blue")
abline(modelo, col = "red", lwd = 2)

# Valores ajustados e resíduos
valores_ajustados <- fitted(modelo)
residuos <- residuals(modelo)
tabela_res <- data.frame(X = dados$horas_estudo, Y = dados$nota, 
                         Y_ajustado = valores_ajustados, Residuo = residuos)
cat("\nPrimeiros 5 registros\n")
print(head(tabela_res, 5))

cat("Soma dos resíduos:", sum(residuos), "\n")
cat("Soma X*resíduos:", sum(dados$horas_estudo * residuos), "\n")

# ANOVA
cat("\nANOVA\n")
print(anova(modelo))

# Intervalos de confiança
cat("\nIntervalos de Confiança (95%)\n")
confint(modelo, level = 0.95)

# Estimação média vs Previsão individual
novos_dados <- data.frame(horas_estudo = c(6, 10))

# Previsão média (Intervalo de confiança)
cat("\nPrevisão da Média (Confiança 95%)\n")
predict(modelo, newdata = novos_dados, interval = "confidence")

# Previsão individual 
cat("\nPrevisão Individual (Predição 95%)\n")
predict(modelo, newdata = novos_dados, interval = "prediction")

# 11. Diagnóstico do Modelo
par(mfrow=c(2,2)) 
plot(modelo)
par(mfrow=c(1,1))