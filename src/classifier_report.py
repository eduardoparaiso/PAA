from collections import Counter
from sklearn.metrics import classification_report
import seaborn as sns

def classificar_com_base_nas_distancias(dist_normal, dist_ami):
  # Inicializa um dicionário vazio para armazenar as classificações por classe
  classificacoes = {}

  # Itera sobre as chaves (classes) presentes no dicionário de distâncias para o grupo "NORMAL"
  for classe, _ in dist_normal.items():
    # Compara as distâncias entre a classe atual e os dois grupos: NORMAL e AMI
    if dist_normal[classe] < dist_ami[classe]:
      classificacoes[classe] = "NORMAL"  # Classifica como "NORMAL" se a distância for menor
    else:
      classificacoes[classe] = "AMI"     # Caso contrário, classifica como "AMI"

  # Retorna o dicionário com as classificações por classe
  return classificacoes

def votacao_final(classificacoes_por_classe):
    # Conta quantas vezes cada classe ("NORMAL" ou "AMI") foi atribuída
    contagem = Counter(classificacoes_por_classe.values())

    # Se houver mais votos para "NORMAL", retorna "NORMAL"
    if contagem["NORMAL"] > contagem["AMI"]:
        return "NORMAL"
    
    # Se houver mais votos para "AMI", retorna "AMI"
    elif contagem["AMI"] > contagem["NORMAL"]:
        return "AMI"
    
    else:
        # Em caso de empate, realiza um critério de desempate baseado na soma das distâncias

        # [Não verificado] Variáveis externas são usadas aqui sem serem passadas como parâmetros
        # Soma das distâncias totais para cada grupo
        soma_normal = sum(dist_normal.values())
        soma_ami = sum(dist_ami.values())

        # Retorna "NORMAL" se a soma das distâncias para o grupo NORMAL for menor; caso contrário, "AMI"
        return "NORMAL" if soma_normal < soma_ami else "AMI"

def matriz_confusao(df):
  # Cria uma matriz de confusão a partir de um DataFrame com colunas 'label' (valor real) e 'predict' (valor previsto)
  matriz_confusao = pd.crosstab(df['label'], df['predict'], rownames=['Atual'], colnames=['Previsto'])

  # Plota a matriz como um mapa de calor (heatmap)
  sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')  # 'annot=True' adiciona os valores, 'fmt=d' formata como inteiros

  # Define o título do gráfico
  plt.title('Matriz de Confusão')

  # Exibe o gráfico
  plt.show()
    
def report(y_test, y_pred):
  # Gera e imprime um relatório de classificação contendo:
  # precisão, recall, f1-score e suporte para cada classe
  # target_names especifica os nomes legíveis das classes 0 e 1
  print(classification_report(y_test, y_pred, target_names=['NORMAL (0)', 'AMI (1)']))