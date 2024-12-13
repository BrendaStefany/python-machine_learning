import gym
import numpy as np
import random
import matplotlib.pyplot as plt  # Para visualização do gráfico

# Criar o ambiente do CartPole
ambiente = gym.make('CartPole-v1', render_mode="human")

# Parâmetros do Q-learning
num_episodios = 10  # Número de episódios para o treinamento
taxa_aprendizado = 0.1  # Taxa de aprendizado para a atualização da Q-table
fator_desconto = 0.99  # Fator de desconto para calcular o valor das recompensas futuras
probabilidade_exploracao = 1.0  # Probabilidade inicial de explorar ações aleatórias
decadencia_exploracao = 0.995  # Taxa de decadência da probabilidade de exploração
min_probabilidade_exploracao = 0.01  # Probabilidade mínima de exploração

# Tabela Q para armazenar os valores Q para cada par estado-ação
tabela_q = np.zeros((1, 1, 1, 1, ambiente.action_space.n))  # Tabela Q com 5 dimensões

# Função de discretização do estado
def discretizar_estado(estado):
    return (0, 0, 0, 0)  # Retorna uma tupla de 4 elementos fixos para simplificar

# Lista para armazenar as recompensas por episódio
recompensas_por_episodio = []

# Treinamento do agente com Q-learning
for episodio in range(num_episodios):
    estado, _ = ambiente.reset()  # Resetando o ambiente e pegando o estado inicial
    terminado = False  # Variável para verificar se o episódio acabou
    recompensa_total = 0  # Acumulador de recompensas no episódio

    while not terminado:
        # Escolher a ação: exploração (aleatória) ou exploração (com base na Q-table)
        if random.uniform(0, 1) < probabilidade_exploracao:
            # Exploração: escolhe uma ação aleatória
            acao = ambiente.action_space.sample()  
        else:
            # Exploração: escolhe a ação com o maior valor na Q-table
            acao = np.argmax(tabela_q[discretizar_estado(estado)])  

        # Executa a ação no ambiente
        proximo_estado, recompensa, terminado, truncado, _ = ambiente.step(acao)
        recompensa_total += recompensa  # Acumula a recompensa do episódio

        # Atualiza a tabela Q usando a fórmula do Q-learning
        tabela_q[discretizar_estado(estado)][acao] += taxa_aprendizado * (
            recompensa + fator_desconto * np.max(tabela_q[discretizar_estado(proximo_estado)]) - tabela_q[discretizar_estado(estado)][acao]
        )

        # Atualiza o estado
        estado = proximo_estado
        terminado = terminado or truncado

    # Salva a recompensa acumulada no episódio
    recompensas_por_episodio.append(recompensa_total)

    # Reduz a probabilidade de exploração a cada episódio
    probabilidade_exploracao = max(min_probabilidade_exploracao, probabilidade_exploracao * decadencia_exploracao)

# Plotando o Gráfico de Recompensa
plt.plot(recompensas_por_episodio)
plt.title("Recompensa Acumulada por Episódio")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Acumulada")
plt.grid()
plt.show()

# Fecha o ambiente
ambiente.close()
