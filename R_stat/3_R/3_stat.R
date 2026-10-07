# Parte 1 – Tempo de atendimento de uma plataforma

tempo <- c(
11.4, 12.8, 13.1, 11.7, 12.5,
13.4, 12.9, 11.8, 12.6, 13.0,
12.2, 13.3, 11.9, 12.7, 13.5,
12.4, 12.8, 13.1, 12.0, 12.9
)

# 1. Conhecendo os dados

length(tempo)
mean(tempo)
sd(tempo)
summary(tempo)

# a) Qual é o tamanho da amostra?

# Resposta: 20

# b) Qual é a média amostral?

# Resposta: 12.75

# c) Qual é o desvio padrão amostral?

# Resposta: 0,5991222

# d) A média observada parece próxima de 12 minutos? Comente brevemente.

# Resposta: A média é superior a 12 minutos

# 2. Visualização dos dados

# Resposta: Os dados aparentam ter uma distribuição razoavelmente simétrica e sem outliers.

hist(
    tempo,
    main = "Tempo de atendimento",
    xlab = "Tempo (minutos)"
)

boxplot(
    tempo,
    horizontal = TRUE,
    main = "Tempo de atendimento"
)

qqnorm(tempo)
qqline(tempo)


# Parte 2 – Formulação do teste

# 3. Parâmetro de interesse

# Resposta: μ = Tempo médio populacional de atendimento dos chamados na plataforma

# 4. Hipóteses estatísticas

# Resposta: H0: μ = 12 H1: μ ≠ 12

# 5. Tipo de teste

# Resposta: Bilateral

# Parte 3 – Realizando o teste no R

resultado <- t.test(
    tempo,
    mu = 12,
    alternative = "two.sided",
    conf.level = 0.95
)
resultado

# 6. Identificação dos resultados

# Resposta:

# Média amostral: 12,6
# Estatística t: 4,4787
# Graus de liberdade: 19
# p-valor: 0,0002592
# Limite inferior do IC de 95%: 12,31951
# Limite superior do IC de 95%: 12,88049

# Parte 4 – Tomando a decisão

# 7. Compare o p-valor com α

# Resposta: p-valor (0,0002592) < 0,05 Portanto: rejeitamos H0

# 8. Conclusão contextual

# Resposta: É possível concluir que o tempo médio é diferente de 12 minutos

# Parte 5 – Relação entre teste e intervalo de confiança

# 9. O valor 12 pertence ao intervalo de confiança de 95%?

# Resposta: Não

# 10. O resultado do intervalo é coerente com a decisão do teste?

# Resposta: É coerente, o valor 12 não é um candidato para a média populacional sob 95% de confiança

# Parte 6 – Mudando a pergunta de pesquisa

# 11. Formule as novas hipóteses

# Resposta: H0: μ = 12 H1: μ > 12

# 12. Execute o teste unilateral

t.test(
    tempo,
    mu = 12,
    alternative = "greater"
)

# Resposta: p-valor = 0,0001296

# 13. Compare os dois testes

# Resposta: O p-valor unilateral é a metade do p-valor bilateral 

# Parte 7 – Teste para uma proporção

n <- 120
sucessos <- 64

phat <- sucessos / n
phat

# 14. Proporção observada

# Resposta:

# 15. Formulação das hipóteses

# Resposta: pˆ = 0,5333

# 16. Execute o teste no R

prop.test(
    x = sucessos,
    n = n,
    p = 0.60,
    alternative = "less",
    correct = FALSE
)

# Resposta: H0: p = 0,60 H1: p < 0,60, unilateral à esquerda

# 17. Conclusão contextual

# Resposta: Não é possível afirmar que a taxa verdadeira de usuários é menor que 60%

# Desafio final – A fronteira de 5%

# Resposta:

# 0,032	Rejeita H0,	Sim
# 0,318	Não rejeita H0,	Não
# 0,049	Rejeita H0,	Sim
# 0,051	Não rejeita H0,	Não

# 18. Discussão

# Resposta: Não, na prática, a força de evidência de ambos é quase idêntica

