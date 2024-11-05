# Bibliotecas
import numpy as np
import random
import itertools
import time

#alterem o codigo para o maximo de conflitos
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


# Gerar a próxima solução (troca os vizinhos)
def gera_proxima_solucao(bestsol, fitbestsol, tabu, n, tempo_freq, registra_tabu_melhor, forcar_conv):
    melhor_fitness = fitbestsol
    melhor_vizinho = None
    movimentos = list(range(len(tabu)))

    random.shuffle(movimentos)  # Otimização para evitar sempre processar na mesma ordem
    for i in movimentos:
        if tabu[i][2] == 0 and tabu[i][3] == 0:
            # Aplica o movimento
            idx1, idx2 = tabu[i][0], tabu[i][1]
            bestsol[idx1], bestsol[idx2] = bestsol[idx2], bestsol[idx1]  # Troca as posições

            # Calcula o fitness do vizinho
            fitness = funobj(bestsol)
            aceita_vizinho = fitness >= melhor_fitness if not forcar_conv else fitness > melhor_fitness

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

# Parametros do algoritmo
n = 50  # Número de rainhas
maxit = 5000 # Número máximo de iterações
tempo = 3  # Tempo máximo de um movimento tabu
tempo_freq = 5  # Frequência máxima de um movimento
registra_tabu_melhor = True  # Registra melhor movimento
forcar_conv = False  # Forçar convergência
it_limite = 300  # Limite de iterações


# --------------------------------------------------------------

# Função para inicializar a solução
def inicia_solucao(n):
    sol = random.sample(range(n), n)
    fit = funobj(sol)
    return sol, fit


# Inicializa uma solução
sol_atual, fit_atual = inicia_solucao(n)

# Inicializa a lista de movimentos possíveis e a matriz tabu
movs_possiveis = list(itertools.combinations(range(n), 2))
tabu = np.array([[mov[0], mov[1], 0, 0] for mov in movs_possiveis], dtype=int)


# Melhor solução encontrada
melhor_sol = sol_atual.copy()
melhor_fit = fit_atual

pior_sol = sol_atual.copy()
pior_fit = fit_atual

# --------------------------------------------------------------

# Inicia o tempo
start_time = time.time()

# Busca Tabu
it = 1
while it <= maxit and fit_atual != 0:
    # Checa o limite iterações
    if it % it_limite == 0:
        sol_atual, fit_atual = inicia_solucao(n)  # Gera nova solução
        tabu = np.array([[mov[0], mov[1], 0, 0] for mov in movs_possiveis], dtype=int)  # Reinicia a matriz tabu
        melhor_sol = sol_atual.copy()
        melhor_fit = fit_atual

    # Gera a próxima solução
    sol_atual = gera_proxima_solucao(sol_atual, fit_atual, tabu, n, tempo_freq, registra_tabu_melhor, forcar_conv)
    fit_atual = funobj(sol_atual)

    # Verifica se encontramos uma melhor solução
    if fit_atual < melhor_fit:
        melhor_sol = sol_atual.copy()
        melhor_fit = fit_atual

    # Atualiza a lista tabu
    atualiza_lista_tabu(tabu)
    it += 1

# --------------------------------------------------------------

# Calcula o tempo total
end_time = time.time()
execution_time = end_time - start_time

# --------------------------------------------------------------

# Exibe a pior solução
print(f"Melhor solução: {pior_sol}")
print(f"Conflitos na melhor solução: {pior_fit}")
print(f"Tempo de execução: {execution_time:.4f} segundos")

