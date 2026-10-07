# Python — Testes de Hipoteses

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Configurações
ALFA = 0.05
FIGURAS = Path(__file__).resolve().parent / "figuras_python"
FIGURAS.mkdir(exist_ok=True)


# Funções auxiliares
def decisao(p_valor: float, alfa: float = ALFA) -> str:
    return "Rejeitar H0" if p_valor <= alfa else "Não rejeitar H0"

def relatorio(nome: str, t: float, gl: float, p: float, ic=None) -> None:
    print(f"\n{'=' * 14} {nome} {'=' * 14}")
    print(f"Estatística t: {t:.6f}")
    print(f"Graus de liberdade: {gl:.6f}")
    print(f"Valor-p: {p:.8g}")
    if ic is not None:
        print(f"Intervalo de confiança de 95%: ({ic.low:.5f}, {ic.high:.5f})")
    print(f"Decisão com α = 0,05: {decisao(p, 0.05)}")
    print(f"Decisão com α = 0,01: {decisao(p, 0.01)}")

def salvar_figura(nome: str) -> None:
    plt.tight_layout()
    destino = FIGURAS / nome
    plt.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figura salva: {destino}")

def grafico_qq(valores, titulo: str, arquivo: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(valores, dist="norm", plot=ax)
    ax.set_title(titulo)
    salvar_figura(arquivo)


# Análises
def exercicio_a_amostras_pareadas():
    """
    H0: média das diferenças (depois - antes) = 0
    H1: média das diferenças (depois - antes) > 0
    """
    antes = np.array([5.2, 6.0, 4.8, 5.5, 6.2, 5.9, 4.5, 6.3,
                      5.1, 5.7, 6.1, 5.4, 4.9, 5.8, 5.3])
    depois = np.array([6.1, 6.8, 5.9, 6.0, 7.1, 6.4, 5.6, 6.9,
                       5.8, 6.5, 6.0, 6.2, 5.7, 6.6, 6.1])

    dados = pd.DataFrame({
        "estudante": np.arange(1, len(antes) + 1),
        "antes": antes,
        "depois": depois,
    })
    dados["diferenca"] = dados["depois"] - dados["antes"]
    diferencas = dados["diferenca"].to_numpy()

    print("\nEXERCÍCIO A — Dados dos estudantes")
    print(dados.to_string(index=False))
    print(f"Média antes: {np.mean(antes):.4f}")
    print(f"Média depois: {np.mean(depois):.4f}")
    print(f"Média das diferenças: {np.mean(diferencas):.4f}")
    print(f"DP amostral das diferenças: {np.std(diferencas, ddof=1):.4f}")

    # Gráficos
    plt.figure(figsize=(6, 4))
    plt.hist(diferencas, bins=6, edgecolor="black")
    plt.xlabel("Nota depois − nota antes")
    plt.ylabel("Frequência")
    plt.title("Distribuição das diferenças")
    salvar_figura("A_histograma_diferencas.png")
    
    plt.figure(figsize=(6, 3))
    plt.boxplot(diferencas, vert=False)
    plt.xlabel("Nota depois − nota antes")
    plt.title("Boxplot das diferenças")
    salvar_figura("A_boxplot_diferencas.png")
    
    grafico_qq(diferencas, "Q-Q plot das diferenças", "A_qqplot_diferencas.png")

    # Testes
    shapiro_a = stats.shapiro(diferencas)
    print(f"Shapiro-Wilk das diferenças: W = {shapiro_a.statistic:.5f}; p = {shapiro_a.pvalue:.6g}")

    res_a = stats.ttest_rel(depois, antes, alternative="greater")
    ic_a = res_a.confidence_interval(confidence_level=0.95)
    relatorio("A: teste t pareado", res_a.statistic, res_a.df, res_a.pvalue, ic_a)

    # Conferência
    n = len(diferencas)
    t_manual = np.mean(diferencas) / (np.std(diferencas, ddof=1) / np.sqrt(n))
    p_manual = stats.t.sf(t_manual, df=n - 1)
    print(f"Conferência manual: t = {t_manual:.6f}; p = {p_manual:.8g}")
    assert np.isclose(t_manual, res_a.statistic)
    assert np.isclose(p_manual, res_a.pvalue)


def exercicio_b_uma_amostra():
    """
    H0: média populacional = 30 minutos
    H1: média populacional != 30 minutos
    """
    tempo = np.array([29, 34, 31, 28, 33, 35, 30, 32, 36, 29, 34, 31])
    print("\nEXERCÍCIO B — Tempo de atendimento")
    print(f"n = {len(tempo)}; média = {np.mean(tempo):.4f}; DP amostral = {np.std(tempo, ddof=1):.4f}")

    # Gráficos
    plt.figure(figsize=(6, 4))
    plt.hist(tempo, bins=6, edgecolor="black")
    plt.xlabel("Tempo de atendimento (minutos)")
    plt.ylabel("Frequência")
    plt.title("Tempos de atendimento")
    salvar_figura("B_histograma_tempo.png")
    
    grafico_qq(tempo, "Q-Q plot dos tempos", "B_qqplot_tempo.png")
    
    # Testes
    shapiro_b = stats.shapiro(tempo)
    print(f"Shapiro-Wilk: W = {shapiro_b.statistic:.5f}; p = {shapiro_b.pvalue:.6g}")

    res_b = stats.ttest_1samp(tempo, popmean=30, alternative="two-sided")
    ic_b = res_b.confidence_interval(confidence_level=0.95)
    relatorio("B: teste t de uma amostra", res_b.statistic, res_b.df, res_b.pvalue, ic_b)
    
    # Conferência
    t_manual_b = (np.mean(tempo) - 30) / (np.std(tempo, ddof=1) / np.sqrt(len(tempo)))
    p_manual_b = 2 * stats.t.sf(abs(t_manual_b), df=len(tempo) - 1)
    print(f"Conferência manual: t = {t_manual_b:.6f}; p = {p_manual_b:.8g}")
    assert np.isclose(t_manual_b, res_b.statistic)
    assert np.isclose(p_manual_b, res_b.pvalue)


def exercicio_c_turmas_independentes():
    """
    H0: média A - média B = 0
    H1: média A - média B != 0
    """
    turma_a = np.array([6.2, 6.8, 7.1, 5.9, 6.5, 7.0, 6.6, 6.3, 7.2, 6.7])
    turma_b = np.array([5.4, 6.1, 5.8, 6.7, 5.5, 6.0, 5.9, 6.2, 5.6, 6.4])
    
    print("\nEXERCÍCIO C — Duas turmas independentes")
    print(f"Média A: {np.mean(turma_a):.4f}; média B: {np.mean(turma_b):.4f}")
    print(f"DP A: {np.std(turma_a, ddof=1):.4f}; DP B: {np.std(turma_b, ddof=1):.4f}")

    # Gráficos
    plt.figure(figsize=(6, 4))
    plt.boxplot([turma_a, turma_b], tick_labels=["Turma A", "Turma B"])
    plt.ylabel("Nota")
    plt.title("Distribuição das notas por turma")
    salvar_figura("C_boxplots_turmas.png")

    # Testes
    print(f"Shapiro turma A: p = {stats.shapiro(turma_a).pvalue:.6g}")
    print(f"Shapiro turma B: p = {stats.shapiro(turma_b).pvalue:.6g}")
    
    res_c = stats.ttest_ind(turma_a, turma_b, equal_var=False, alternative="two-sided")
    ic_c = res_c.confidence_interval(confidence_level=0.95)
    relatorio("C: teste t de Welch", res_c.statistic, res_c.df, res_c.pvalue, ic_c)

    # Conferência
    na, nb = len(turma_a), len(turma_b)
    va, vb = np.var(turma_a, ddof=1), np.var(turma_b, ddof=1)
    se2 = va / na + vb / nb
    t_welch_manual = (np.mean(turma_a) - np.mean(turma_b)) / np.sqrt(se2)
    gl_welch_manual = se2**2 / ((va / na)**2 / (na - 1) + (vb / nb)**2 / (nb - 1))
    p_welch_manual = 2 * stats.t.sf(abs(t_welch_manual), df=gl_welch_manual)
    
    print(f"Conferência manual: t = {t_welch_manual:.6f}; gl = {gl_welch_manual:.6f}; p = {p_welch_manual:.8g}")
    assert np.isclose(t_welch_manual, res_c.statistic)
    assert np.isclose(gl_welch_manual, res_c.df)
    assert np.isclose(p_welch_manual, res_c.pvalue)


def exibir_observacoes():
    print("\nOBSERVAÇÕES DIDÁTICAS")
    print("- Verifique independência e desenho amostral, não somente normalidade.")
    print("- Valor-p não representa a probabilidade de H0 ser verdadeira.")
    print("- Sem grupo de controle, a comparação antes-depois não estabelece causalidade.")
    print("- Dados fornecidos são fictícios e servem ao aprendizado.")
    print("\nTodos os testes foram conferidos por cálculo independente.")


# Execução
def main():
    exercicio_a_amostras_pareadas()
    exercicio_b_uma_amostra()
    exercicio_c_turmas_independentes()
    exibir_observacoes()

if __name__ == "__main__":
    main()