# Bibliotecas
import numpy as np
import random
import itertools
import time

# --------------------------------------------------------------
#Sem tipo analise e sem plotagem

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

# Gerar a próxima solução (troca os vizinhos)
def gera_proxima_solucao(bestsol, fitbestsol, tabu, n, tempofreq, registraTabuMelhor, forcarConv):
    vizinhos = []
    melhor_fitness = fitbestsol
    melhor_vizinho = bestsol

    # Movimentos fora do tabu
    ind_mov_possiveis = [i for i in range(len(tabu)) if tabu[i][2] == 0 and tabu[i][3] == 0]

    for i in ind_mov_possiveis:
        vizinho = bestsol.copy()
        vizinho[tabu[i][0]], vizinho[tabu[i][1]] = vizinho[tabu[i][1]], vizinho[tabu[i][0]]

        fitness = funobj(vizinho)

        aceita_vizinho = fitness <= melhor_fitness if not forcarConv else fitness < melhor_fitness

        if aceita_vizinho:
            melhor_fitness = fitness
            melhor_vizinho = vizinho.copy()
            if registraTabuMelhor:
                tabu[i][2] = tempo
                tabu[i][3] += 1
            break

        vizinhos.append((vizinho, fitness))

    if melhor_vizinho is bestsol and vizinhos:
        melhor_vizinho, melhor_fitness = min(vizinhos, key=lambda x: x[1])

        mov = [i for i in range(n) if melhor_vizinho[i] != bestsol[i]]
        ind_mov_tabu = next((i for i in range(len(tabu)) if {tabu[i][0], tabu[i][1]} == set(mov)), None)

        if ind_mov_tabu is not None:
            tabu[ind_mov_tabu][2] = tempo
            tabu[ind_mov_tabu][3] += 1

    return melhor_vizinho

# Atualizar a lista tabu
def atualiza_lista_tabu(tabu):
    tabu[:, 2] = np.maximum(tabu[:, 2] - 1, 0)

# --------------------------------------------------------------

# Parâmetros do algoritmo
n = 100  # Número de rainhas
maxit = 1000  # Número máximo de iterações
tempo = 5  # Tempo máximo de um movimento tabu
tempofreq = 5  # Frequência máxima de um movimento
registraTabuMelhor = False  # Registra melhor movimento
forcarConv = True # Forçar convergência
reiniciar_limite = 500  # Limite de iterações

# Função para inicializar a solução
def inicializa_solucao(n):
    sol = random.sample(range(n), n)
    fit = funobj(sol)
    return sol, fit

# Inicia um vetor aleatório
sol_atual = random.sample(range(n), n)
fit_atual = funobj(sol_atual)

# Inicializa a lista de movimentos possíveis e a matriz tabu
movs_possiveis = list(itertools.combinations(range(n), 2))
tabu = np.array([[mov[0], mov[1], 0, 0] for mov in movs_possiveis], dtype=int)

# Melhor solução encontrada
best_ind = sol_atual.copy()
best_fit_ind = fit_atual

# Inicia o tempo
start_time = time.time()

# Busca Tabu
it = 1
while it <= maxit and fit_atual != 0:
    if it % reiniciar_limite == 0:  # Checa se atingiu limite iterações
        sol_atual, fit_atual = inicializa_solucao(n)  # Gera nova solução
        tabu = np.array([[mov[0], mov[1], 0, 0] for mov in movs_possiveis], dtype=int)  # Reinicia a matriz tabu
        best_ind = sol_atual.copy()
        best_fit_ind = fit_atual

    sol_atual = gera_proxima_solucao(sol_atual, fit_atual, tabu, n, tempofreq, registraTabuMelhor, forcarConv)
    fit_atual = funobj(sol_atual)

    if fit_atual < best_fit_ind:
        best_ind = sol_atual.copy()
        best_fit_ind = fit_atual

    atualiza_lista_tabu(tabu)
    it += 1

# Calcula o tempo total
end_time = time.time()
execution_time = end_time - start_time

# --------------------------------------------------------------

# Exibe a melhor solução
print(f"Melhor solução: {best_ind}")
print(f"Conflitos na melhor solução: {best_fit_ind}")
print(f"Tempo de execução: {execution_time:.4f} segundos")
