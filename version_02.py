# Bibliotecas
import numpy as np
import random
import itertools
import time


# --------------------------------------------------------------

# Função objetivo (achar conflitos nas diagonais)
def funobj(sol):
    n = len(sol)
    dn = np.zeros(2 * n - 1, dtype=int)  # Indica o tipo de dado
    dp = np.zeros(2 * n - 1, dtype=int)

    np.add.at(dn, np.arange(n) + sol, 1)
    np.add.at(dp, np.arange(n) - sol + (n - 1), 1)

    # Conta o número de conflitos
    conflitos = np.sum(dn[dn > 1] - 1) + np.sum(dp[dp > 1] - 1)
    return conflitos


# Modifique a função de geração da próxima solução para respeitar a posição fixa
def gera_proxima_solucao(bestsol, fitbestsol, tabu, n, tempo_freq, registra_tabu_melhor, forcar_conv, posicao_fixa):
    melhor_fitness = fitbestsol
    melhor_vizinho = None
    movimentos = list(range(len(tabu)))

    random.shuffle(movimentos)  # Otimização para evitar sempre processar na mesma ordem
    for i in movimentos:
        idx1, idx2 = tabu[i][0], tabu[i][1]

        # Verifica se uma das posições é a posição fixa; se sim, ignora esse movimento
        if posicao_fixa in (idx1, idx2) or tabu[i][2] != 0 or tabu[i][3] != 0:
            continue

        # Aplica o movimento
        bestsol[idx1], bestsol[idx2] = bestsol[idx2], bestsol[idx1]

        # Calcula o fitness do vizinho
        fitness = funobj(bestsol)
        aceita_vizinho = fitness <= melhor_fitness if not forcar_conv else fitness < melhor_fitness

        if aceita_vizinho:
            melhor_fitness = fitness
            melhor_vizinho = bestsol.copy()

            if registra_tabu_melhor:
                tabu[i][2] = tempo
                tabu[i][3] += 1
            break  # Aceitamos esse vizinho, saímos do loop

        # Reverte o movimento
        bestsol[idx1], bestsol[idx2] = bestsol[idx2], bestsol[idx1]

    if melhor_vizinho is None:
        # Se nenhum vizinho foi melhor, escolha o melhor dos piores
        for i in movimentos:
            idx1, idx2 = tabu[i][0], tabu[i][1]

            if posicao_fixa in (idx1, idx2):
                continue

            bestsol[idx1], bestsol[idx2] = bestsol[idx2], bestsol[idx1]
            fitness = funobj(bestsol)

            if fitness < melhor_fitness:
                melhor_fitness = fitness
                melhor_vizinho = bestsol.copy()

            bestsol[idx1], bestsol[idx2] = bestsol[idx2], bestsol[idx1]

    return melhor_vizinho if melhor_vizinho is not None else bestsol


# Atualizar a lista tabu
def atualiza_lista_tabu(tabu):
    tabu[:, 2] = np.maximum(tabu[:, 2] - 1, 0)


# --------------------------------------------------------------

# Parâmetros do algoritmo
n = 100  # Número de rainhas
maxit = 1000  # Número máximo de iterações
tempo = 3  # Tempo máximo de um movimento tabu
tempo_freq = 5  # Frequência máxima de um movimento
registra_tabu_melhor = True  # Registra melhor movimento
forcar_conv = False  # Forçar convergência
it_limite = 300  # Limite de iterações

posicao_fixa = 5  # Defina a posição da rainha fixa


# --------------------------------------------------------------

# Função para inicializar a solução
def inicia_solucao(n):
    sol = random.sample(range(n), n)
    fit = funobj(sol)
    return sol, fit


# Inicializa uma solução
sol_atual, fit_atual = inicia_solucao(n)

print(sol_atual)
print("Posição fixa: {0}".format(posicao_fixa))

# Inicializa a lista de movimentos possíveis e a matriz tabu
movs_possiveis = list(itertools.combinations(range(n), 2))
tabu = np.array([[mov[0], mov[1], 0, 0] for mov in movs_possiveis], dtype=int)

# Melhor solução encontrada
melhor_sol = sol_atual.copy()
melhor_fit = fit_atual

# --------------------------------------------------------------

# Inicia o tempo
start_time = time.time()

# Busca Tabu
it = 1
while it <= maxit and fit_atual != 0:
    if it % it_limite == 0:
        sol_atual, fit_atual = inicia_solucao(n)
        tabu = np.array([[mov[0], mov[1], 0, 0] for mov in movs_possiveis], dtype=int)
        melhor_sol = sol_atual.copy()
        melhor_fit = fit_atual

    sol_atual = gera_proxima_solucao(sol_atual, fit_atual, tabu, n, tempo_freq, registra_tabu_melhor, forcar_conv, posicao_fixa)
    fit_atual = funobj(sol_atual)

    if fit_atual < melhor_fit:
        melhor_sol = sol_atual.copy()
        melhor_fit = fit_atual

    atualiza_lista_tabu(tabu)
    it += 1


# --------------------------------------------------------------

# Calcula o tempo total
end_time = time.time()
execution_time = end_time - start_time

# --------------------------------------------------------------

# Exibe a melhor solução
print(f"Melhor solução: {melhor_sol}")
print(f"Conflitos na melhor solução: {melhor_fit}")
print(f"Tempo de execução: {execution_time:.4f} segundos")
