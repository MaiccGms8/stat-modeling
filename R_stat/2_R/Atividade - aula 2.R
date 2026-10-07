# R — Testes de Hipoteses

alpha <- 0.05

interpretar <- function(resultado, nome, alpha=0.05) {
  cat("\n=", nome, "=\n")
  print(resultado)
  cat("\nEstatistica:", unname(resultado$statistic),
      "\nGraus de liberdade:", unname(resultado$parameter),
      "\nValor-p:", resultado$p.value,
      "\nDecisao:", if (resultado$p.value <= alpha)
        "Rejeitar H0" else "Nao rejeitar H0", "\n")
}

# A - MESMOS ESTUDANTES, ANTES E DEPOIS
antes <- c(5.2, 6.0, 4.8, 5.5, 6.2, 5.9, 4.5, 6.3,
           5.1, 5.7, 6.1, 5.4, 4.9, 5.8, 5.3)
depois <- c(6.1, 6.8, 5.9, 6.0, 7.1, 6.4, 5.6, 6.9,
            5.8, 6.5, 6.0, 6.2, 5.7, 6.6, 6.1)
dados <- data.frame(estudante=seq_along(antes), antes=antes, depois=depois)
dados$diferenca <- dados$depois - dados$antes
print(dados)
cat("\nMedia antes:", mean(antes),
    "\nMedia depois:", mean(depois),
    "\nMedia diferencas:", mean(dados$diferenca),
    "\nDP diferencas:", sd(dados$diferenca), "\n")

hist(dados$diferenca, breaks=6, col="lightblue",
     main="Diferencas individuais", xlab="Depois - antes")
boxplot(dados$diferenca, horizontal=TRUE, main="Diferencas")
qqnorm(dados$diferenca); qqline(dados$diferenca, col="red")
print(shapiro.test(dados$diferenca)) # nao prova normalidade

# H0: mu_d = 0; H1: mu_d > 0
res_A <- t.test(depois, antes, paired=TRUE,
                alternative="greater", mu=0, conf.level=0.95)
interpretar(res_A, "A: t pareado", alpha)

n <- length(dados$diferenca)
t_manual <- mean(dados$diferenca)/(sd(dados$diferenca)/sqrt(n))
p_manual <- pt(t_manual, df=n-1, lower.tail=FALSE)
cat("t manual =", t_manual, "; p manual =", p_manual, "\n")
stopifnot(isTRUE(all.equal(unname(res_A$statistic), t_manual)),
          isTRUE(all.equal(res_A$p.value, p_manual)))
cat("Em alfa=0.01:", if (res_A$p.value <= 0.01)
    "Rejeitar H0" else "Nao rejeitar H0", "\n")

# B - UMA AMOSTRA
# H0: mu = 30; H1: mu != 30
tempo <- c(29, 34, 31, 28, 33, 35, 30, 32, 36, 29, 34, 31)
cat("\nTempo medio:", mean(tempo), "\n")
hist(tempo, col="lightgray", main="Tempo de atendimento", xlab="Minutos")
qqnorm(tempo); qqline(tempo, col="red")
print(shapiro.test(tempo))
res_B <- t.test(tempo, mu=30, alternative="two.sided", conf.level=0.95)
interpretar(res_B, "B: uma amostra", alpha)

# C - DOIS GRUPOS INDEPENDENTES (WELCH) 
# H0: mu_A - mu_B = 0; H1: mu_A - mu_B != 0
turma_A <- c(6.2, 6.8, 7.1, 5.9, 6.5, 7.0, 6.6, 6.3, 7.2, 6.7)
turma_B <- c(5.4, 6.1, 5.8, 6.7, 5.5, 6.0, 5.9, 6.2, 5.6, 6.4)
cat("\nMedia turma A:", mean(turma_A),
    "\nMedia turma B:", mean(turma_B), "\n")
boxplot(turma_A, turma_B, names=c("A", "B"),
        col=c("lightblue", "lightgray"), ylab="Nota")
res_C <- t.test(turma_A, turma_B, alternative="two.sided",
                var.equal=FALSE, conf.level=0.95)
interpretar(res_C, "C: Welch independente", alpha)

cat("\nObservacao: verificar independencia e pressupostos pelo desenho,\n",
    "complementando com graficos. Valor-p nao e P(H0 verdadeira).\n",
    "Resultados antes-depois nao demonstram causalidade sem controle.\n")
