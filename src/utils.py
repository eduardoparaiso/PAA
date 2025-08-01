from matplotlib import pyplot as plt
import seaborn as sns

def plot_ECG(ECG, derivacoes=None, fs=500, figsize=(12, 4)):
  # Cria o vetor de tempo em segundos, baseado no número de amostras e na frequência de amostragem fs
  tempo = np.arange(ECG.shape[0]) / fs  # eixo X em segundos

  # Se forem especificadas derivações para plotar
  if derivacoes is not None:
    # Para cada derivação indicada, cria um gráfico separado
    for i in derivacoes:
      plt.figure(figsize=figsize)          # Define o tamanho da figura
      plt.plot(tempo, ECG[:, i])           # Plota o sinal da derivação i em função do tempo
      plt.title(f'Derivação {i + 1}')     # Título do gráfico indicando a derivação (index + 1 para começar em 1)
      plt.xlabel('Tempo (s)')              # Label do eixo X (tempo em segundos)
      plt.ylabel('Amplitude (mV)')         # Label do eixo Y (amplitude em milivolts)
      plt.grid(True)                       # Ativa a grade no gráfico para melhor visualização
      plt.tight_layout()                   # Ajusta o layout para não sobrepor elementos
      plt.show()                          # Exibe o gráfico
  else:
    # Caso não sejam especificadas derivações, plota o ECG completo (todas as derivações juntas)
    plt.figure(figsize=figsize)            # Define o tamanho da figura
    plt.plot(tempo, ECG)                   # Plota todas as derivações do ECG em função do tempo
    plt.title('Primeiro ECG')              # Título do gráfico genérico
    plt.xlabel('Tempo (s)')                # Label do eixo X (tempo em segundos)
    plt.ylabel('Amplitude (mV)')           # Label do eixo Y (amplitude em milivolts)
    plt.grid(True)                         # Ativa a grade no gráfico para melhor visualização

def plot_comparacao_ECGs(ECG1, ECG2, derivacoes, fs=500, figsize=(12, 4)):
  # Verifica se os dois sinais ECG têm a mesma forma (mesmo número de amostras e derivações)
  if ECG1.shape != ECG2.shape:
    raise ValueError("ECG1 e ECG2 devem ter a mesma forma (n_amostras x n_derivações)")

  # Cria o vetor de tempo em segundos para o eixo X, baseado no número de amostras e frequência de amostragem fs
  tempo = np.arange(ECG1.shape[0]) / fs

  # Para cada derivação especificada na lista derivacoes
  for i in derivacoes:
    plt.figure(figsize=figsize)                  # Define o tamanho da figura para o gráfico
    plt.plot(tempo, ECG1[:, i], label='ECG 1', color='blue')  # Plota o ECG1 na derivação i em azul
    plt.plot(tempo, ECG2[:, i], label='ECG 2', color='red')   # Plota o ECG2 na derivação i em vermelho
    plt.title(f'Derivação {i}')                   # Define o título do gráfico com a derivação atual
    plt.xlabel('Tempo (s)')                       # Label do eixo X (tempo em segundos)
    plt.ylabel('Amplitude (mV)')                  # Label do eixo Y (amplitude em milivolts)
    plt.legend()                                  # Exibe a legenda para diferenciar os ECGs
    plt.grid(True)                                # Ativa a grade no gráfico para melhor visualização
    plt.tight_layout()                            # Ajusta o layout para evitar sobreposição
    plt.show()                                   # Exibe o gráfico

def plot_beats(ECG_beats, figsize=(4, 6), derivacao=None):
  # Cria uma figura com o tamanho especificado
  plt.figure(figsize=figsize)

  # Cria um subplot na posição 1 de uma grade 2x1 (duas linhas, uma coluna)
  plt.subplot(2, 1, 1)

  # Se uma derivação específica for passada, plota somente essa derivação
  if derivacao is not None:
    plt.plot(ECG[:, derivacao])  # Plota a derivação selecionada do ECG
  else:
    plt.plot(ECG)  # Caso contrário, plota o ECG completo

  # Define o título do gráfico
  plt.title('Primeiro ECG')
  
  # Label do eixo X (número de amostras)
  plt.xlabel('Amostras')
  
  # Label do eixo Y (amplitude em milivolts)
  plt.ylabel('Amplitude (mV)')
  
  # Exibe o gráfico
  plt.show()

def analisa_tempo(tempo_list):
  # Calcula e imprime o tempo médio e o desvio padrão da lista de tempos fornecida
  # np.mean calcula a média dos valores em tempo_list
  # np.std calcula o desvio padrão dos valores em tempo_list
  # A saída é formatada para mostrar 4 casas decimais
  print(f'Tempo médio: {np.mean(tempo_list):.4f} +- {np.std(tempo_list):.4f}')