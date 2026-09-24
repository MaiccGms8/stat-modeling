# Problema: Quais fatores estão associados ao desempenho dos estudantes em uma avaliação

# 1. Base de dados simulada

# Carregua o pacote dplyr
library(dplyr)

set.seed(123)
n <- 80

dados <- tibble::tibble(
    aluno = 1:n,
    grupo = sample(c("Tradicional", "Projeto"), n, TRUE),
    horas_estudo = round(rnorm(n, mean = 6, sd = 2), 1),
    frequencia = round(runif(n, 70, 100), 1),
    nota_final = 45 + 4.2 * horas_estudo + 0.25 * frequencia + 
        ifelse(grupo == "Projeto", 4, 0) + rnorm(n, 0, 7)
)

dados <- dados |>
 mutate(
    horas_estudo = pmax(horas_estudo, 0),
    nota_final = pmin(pmax(nota_final, 0), 100)
 )

# 2. Análise exploratória

# Carregua o pacote tidyverse
library(tidyverse)

# Distribuição e o relacionamento entre variáveis
dados |> 
    summarise(
        media_nota = mean(nota_final),
        dp_nota = sd(nota_final),
        media_estudo = mean(horas_estudo),
        cor_estudo_nota = cor(horas_estudo, nota_final)
    )

dados |> 
    count(grupo)

ggplot(dados, aes(x = horas_estudo, y = nota_final, color = grupo)) +
    geom_point(size = 2, alpha = 0.8) +
    geom_smooth(method = "lm", se = TRUE) +
    labs(x = "Horas de estudo", y = "Nota final", color = "Grupo")

# 3. Intervalo de confiança

# Estimando a média da nota final com incerteza
media <- mean(dados$nota_final)
dp <- sd(dados$nota_final)
n <- nrow(dados)
erro <- qt(0.975, df = n - 1) * dp / sqrt(n)

ic_media <- tibble(
    media = media,
    limite_inferior = media - erro, 
    limite_superior = media + erro
)

ic_media 

# O intervalo indica uma faixa plausível para a média populacional da nota final, sob as suposições do método

# 4. Teste de hipótese

# Comparando a média da nota entre dois grupos, H0 = médias iguais e H1 = médias diferentes 
teste_grupo <- t.test(nota_final ~ grupo, data = dados)
teste_grupo

dados |> 
    group_by(grupo) |>
    summarise(
        media = mean(nota_final),
        dp = sd(nota_final),
        n = n(),
        .groups = "drop"
        )

# 5. Regressão linear simples

# Modelando nota final a partir das horas de estudo
modelo_1 <- lm(nota_final ~ horas_estudo, data = dados)
summary(modelo_1)

broom::tidy(modelo_1, conf.int = TRUE)
broom::glance(modelo_1)

ggplot(dados, aes(horas_estudo, nota_final)) + 
    geom_point(alpha = 0.8) +
    geom_smooth(method = "lm", se = TRUE) + 
    labs(x = "Horas de estudo", y = "Nota final")

# 6. Modelo com mais de uma variável

# Considerando frequência e grupo no mesmo modelo
modelo_2 <- lm(nota_final ~ horas_estudo + frequencia + grupo, data = dados)

summary(modelo_2)
broom::tidy(modelo_2, conf.int = TRUE)
broom::glance(modelo_2)

anova(modelo_1, modelo_2)

# 7. Diagnóstico do modelo 

# Verificação das suposições
par(mfrow = c(2,2))
plot(modelo_2)
par(mfrow = c(1,1))