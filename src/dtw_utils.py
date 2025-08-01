from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from collections import deque

def DTW(s, t):
    # Obtém os comprimentos das duas sequências
    n, m = len(s), len(t)

    # Inicializa a matriz DTW com infinito, com tamanho (n+1) x (m+1)
    # A posição [0, 0] é inicializada com zero para iniciar o cálculo
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0

    # Preenche a matriz DTW com os custos acumulados
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (s[i-1] - t[j-1])**2  # Calcula o custo local: diferença ao quadrado entre os elementos
            
            # Atualiza a célula com o custo local + menor custo acumulado entre inserção, deleção e correspondência
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Inserção (vertical)
                dtw_matrix[i, j-1],    # Deleção (horizontal)
                dtw_matrix[i-1, j-1]   # Correspondência (diagonal)
            )

    # A distância DTW final é a raiz quadrada do custo acumulado no canto inferior direito
    dtw_distance = np.sqrt(dtw_matrix[n, m])
    
    return dtw_distance

def euclidean(a, b):
    # Calcula a distância Euclidiana entre dois vetores unidimensionais
    return np.linalg.norm(np.array(a) - np.array(b))

def dtw(x, y, window=None):
    # Implementação de DTW com restrição de janela (Sakoe-Chiba Band)
    len_x, len_y = len(x), len(y)
    
    # Define o tamanho da janela, se não for especificado
    window = max(window or max(len_x, len_y), abs(len_x - len_y))
    
    dtw_matrix = {}

    # Inicializa a matriz DTW com infinito
    for i in range(-1, len_x):
        for j in range(-1, len_y):
            dtw_matrix[(i, j)] = float('inf')
    dtw_matrix[(-1, -1)] = 0  # condição inicial

    # Preenche a matriz dentro da janela de restrição
    for i in range(len_x):
        for j in range(max(0, i - window), min(len_y, i + window + 1)):
            cost = euclidean(x[i], y[j])
            dtw_matrix[(i, j)] = cost + min(
                dtw_matrix[(i-1, j)],     # inserção
                dtw_matrix[(i, j-1)],     # deleção
                dtw_matrix[(i-1, j-1)]    # correspondência
            )
    return dtw_matrix[(len_x - 1, len_y - 1)]

def expand_window(path, len_x, len_y, radius):
    # Expande o caminho de alinhamento para uma janela no espaço original
    path_set = set(path)
    window = set()

    for i, j in path:
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                ii, jj = 2 * i + a, 2 * j + b  # projeta para escala maior
                if 0 <= ii < len_x and 0 <= jj < len_y:
                    window.add((ii, jj))
    return window

def constrained_dtw(x, y, window):
    # Calcula o DTW restrito à janela especificada
    dtw_matrix = {}
    for i, j in window:
        dtw_matrix[(i, j)] = float('inf')
    dtw_matrix[(-1, -1)] = 0

    # Ordena e percorre os pares da janela para preencher a matriz DTW
    for i, j in sorted(window):
        cost = euclidean(x[i], y[j])
        dtw_matrix[(i, j)] = cost + min(
            dtw_matrix.get((i-1, j), float('inf')),
            dtw_matrix.get((i, j-1), float('inf')),
            dtw_matrix.get((i-1, j-1), float('inf'))
        )

    return dtw_matrix[(len(x)-1, len(y)-1)]


def constrained_dtw(x, y, window):
    dtw_matrix = {}
    for i, j in window:
        dtw_matrix[(i, j)] = float('inf')
    dtw_matrix[(-1, -1)] = 0

    for i, j in sorted(window):
        cost = euclidean(x[i], y[j])
        dtw_matrix[(i, j)] = cost + min(
            dtw_matrix.get((i-1, j), float('inf')),
            dtw_matrix.get((i, j-1), float('inf')),
            dtw_matrix.get((i-1, j-1), float('inf'))
        )

    return dtw_matrix[(len(x)-1, len(y)-1)]

def fastdtw_custom(x, y, radius=1):
    # Implementação recursiva do FastDTW com raio de restrição
    min_time_size = radius + 2

    # Se as séries forem curtas, usa DTW exato com janela
    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, window=radius)

    # Reduz as séries pela metade
    x_shrink = reduce_by_half(x)
    y_shrink = reduce_by_half(y)

    # Aplica FastDTW recursivamente nas versões reduzidas
    distance = fastdtw_custom(x_shrink, y_shrink, radius=radius)

    # Expande a janela com base no caminho presumido (diagonal)
    window = expand_window(
        path=[(i, i) for i in range(min(len(x_shrink), len(y_shrink)))],
        len_x=len(x),
        len_y=len(y),
        radius=radius
    )

    # Executa DTW restrito à janela expandida
    return constrained_dtw(x, y, window)


def calcular_distancias_dtw(batimentos, prototipos, fastDTW=False):
    # Inicialização de variáveis
    inicio = 0        # Marca o início da contagem de tempo para cada par
    fim = 0           # Marca o fim da contagem
    distancias = {}   # Dicionário para armazenar as distâncias DTW por classe
    tempo = []        # Lista para armazenar o tempo de execução por comparação

    # Itera pelas classes presentes no dicionário de batimentos
    for classe in batimentos:
        # Garante que exista um protótipo correspondente para a classe atual
        if classe in prototipos:
            # Se fastDTW for True, usa a versão aproximada (FastDTW)
            if fastDTW:
                # Converte os sinais para arrays numpy colunares (necessário para multivariada)
                bat = np.asarray(batimentos[classe]).reshape(-1, 1)
                prot = np.asarray(prototipos[classe]).reshape(-1, 1)

                # Mede o tempo de execução
                inicio = time.time()
                dist = fastdtw_custom(bat, prot, radius=2)  # Usa FastDTW com raio 2
                fim = time.time()

                distancias[classe] = dist     # Armazena a distância
                tempo.append(fim - inicio)    # Armazena o tempo de execução

            else:
                # Mede o tempo de execução para o DTW exato
                inicio = time.time()
                dist = DTW(batimentos[classe], prototipos[classe])
                fim = time.time()

                distancias[classe] = dist     # Armazena a distância
                tempo.append(fim - inicio)    # Armazena o tempo de execução

    # Retorna o dicionário de distâncias e o tempo médio por comparação
    return distancias, np.mean(tempo)