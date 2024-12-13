# Reinforcement Learning Project

Este repositório contém o código desenvolvido para um projeto de **Aprendizado por Reforço (Reinforcement Learning)**. O objetivo é implementar e experimentar algoritmos que permitem a um agente aprender interagindo com o ambiente.

---

## 📋 Descrição do Projeto

O projeto utiliza aprendizado por reforço para resolver problemas de decisão sequencial. Ele emprega algoritmos como **Q-Learning**, **SARSA**, ou métodos baseados em **Deep Q-Networks (DQN)** para treinar um agente que toma decisões otimizadas com base em recompensas acumuladas.

O ambiente simula o equilíbrio de um pêndulo invertido montado sobre um carrinho em movimento, onde o objetivo é aplicar forças(horizontais) ao carrinho para manter o pêndulo em pé pelo maior tempo possível.

---

## 🛠 Tecnologias Utilizadas

As principais tecnologias e bibliotecas usadas neste projeto incluem:

- **Python 3.1**
- **NumPy** - Operações numéricas e matrizes
- **OpenAI Gym** - Ambientes de simulação

---

## 🔧 Como Funciona

1. **Definição do Ambiente:**
   O código inicializa o ambiente "CartPole-v1" usando OpenAI Gym.

2. **Treinamento do Agente:**
   O agente aprende interagindo com o ambiente usando um dos algoritmos implementados:
   - Escolha de ações com base em uma política.
   - Atualização de valores usando a equação de Bellman.

---

## 📊 Resultados

- **Desempenho Final:** O agente consegue equilibrar o pêndulo, impedindo-o de cair, completando assim o objetivo do jogo.

---

## 🤝 Grupo

- Amanda Francelina da Silva - 01601710
- Brenda Stefany Lima Cavalcanti - 01589516
- Juliana Marinho Xavier - 01132389


---

*Happy Coding! 🚀*
