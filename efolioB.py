#
#============================================================================
# UC: 21071 - Introdução à Inteligência Artificial - 03 - UAb
# e-fólio B  2022-23 
#
# Aluno: 2100927 - Ivo Baptista
# Name        : efolioB.py
# Author      : Ivo Baptista
# Version     : 3.2
# Copyright   : Ivo copyright 
# Description : Problema da Cidade Muralhada Medieval em Python
#===========================================================================
#

# Importando as bibliotecas necessárias
import streamlit as st
import numpy as np
import pandas as pd
from collections import namedtuple
from queue import Queue
from math import exp, sqrt
import matplotlib.pyplot as plt
import time 
from collections import deque
import random
import csv
import os
import base64

# Definindo uma tupla nomeada para representar a posição de uma célula na matriz
Position = namedtuple("Position", ["row", "col"])

# Definindo a classe de Estado, que representa um estado específico do problema
class State:
    def __init__(self, mapa, cost, moves=0):
        self.mapa = np.array(mapa)
        self.moved = np.zeros(self.mapa.shape, dtype=bool)
        self.cost = cost
        self.moves = moves
        
# Função para verificar se uma posição (linha, coluna) é válida na matriz
def get_neighbors(mapa, i, j):
    neighbors = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i < len(mapa) - 1:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j < len(mapa[0]) - 1:
        neighbors.append((i, j + 1))
    return neighbors

# Verifica se uma posição (linha, coluna) é válida na matriz
def isValidPosition(mapa, row, col):
    numRows, numCols = mapa.shape
    return (row >= 0 and row < numRows and col >= 0 and col < numCols)

# Verifica se existe um caminho entre duas posições
def path_exists(mapa, start, end):
    visited = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        visited.add(current)

        if current == end:
            return True

        for neighbor in get_neighbors(mapa, *current):
            if mapa[neighbor[0]][neighbor[1]] != -1 and neighbor not in visited:
                queue.append(neighbor)

    return False

# Aqui é onde a maior parte da lógica do problema é implementada.
# Esta função retorna uma lista de todas as casas que estão conectadas a uma porta específica
def getConnectedHouses(mapa, door):
    connectedHouses = []
    visited = np.zeros(mapa.shape, dtype=bool)
    q = Queue()
    q.put(door)
    visited[door.row][door.col] = True

    dr = [-1, 1, 0, 0, -1, -1, 1, 1]
    dc = [0, 0, -1, 1, -1, 1, -1, 1]

    while not q.empty():
        curr = q.get()
        if mapa[curr.row][curr.col] == 1:
            connectedHouses.append(curr)

        for i in range(8):
            newRow = curr.row + dr[i]
            newCol = curr.col + dc[i]

            if isValidPosition(mapa, newRow, newCol) and not visited[newRow][newCol]:
                if mapa[newRow][newCol] != -1:
                    q.put(Position(newRow, newCol))
                    visited[newRow][newCol] = True

    return connectedHouses

# Aqui é onde o estado é alterado para produzir um novo estado.
# Esta função seleciona aleatoriamente uma casa e a move para um local vazio adjacente, se houver um.
def generateNeighbor(state, doors):
    newState = State(state.mapa.copy(), state.cost, state.moves)
    newState.moves += 1
    numRows, numCols = state.mapa.shape
    dr = [-1, 1, 0, 0, -1, -1, 1, 1]
    dc = [0, 0, -1, 1, -1, 1, -1, 1]

    while True:
        row, col = np.where(state.mapa == 1)
        idx = np.random.choice(len(row))
        house = Position(row[idx], col[idx])

        for i in np.random.permutation(8):
            newRow = house.row + dr[i]
            newCol = house.col + dc[i]
            #print("New row:", newRow, "New col:", newCol)
            if isValidPosition(state.mapa, newRow, newCol) and not state.moved[newRow][newCol] and state.mapa[newRow][newCol] == 0:
                newState.mapa[house.row][house.col] = 0
                newState.mapa[newRow][newCol] = 1
                newState.moved[house.row][house.col] = True
                
                newCost = 0
                for door in doors:  # Iterate over each door
                    connectedHouses = getConnectedHouses(newState.mapa, door)
                    for house in connectedHouses:
                        minDist = float('inf')
                        dist = sqrt((house.row-door.row)**2 + (house.col-door.col)**2)
                        minDist = min(minDist, dist)
                    newCost += minDist
                #print("Old cost:", newState.cost)
                newState.cost = newCost
                #print("New cost:", newState.cost)
                return newState
        
        if np.all(state.moved):
            return None

# Algoritmo de Hill Climbing com número limitado de iterações
def hill_climbing(state, doors, iterations):
    current_state = state
    current_cost = state.cost
    best_state = current_state
    best_cost = current_cost
    generations = 0
    evaluations = 0

    for _ in range(iterations):
        neighbor = generateNeighbor(current_state, doors)
        evaluations += 1

        if neighbor is None or neighbor.cost > current_cost:
            return best_state, best_cost, generations, evaluations

        current_state = neighbor
        current_cost = neighbor.cost

        if current_cost < best_cost:
            best_state = current_state
            best_cost = current_cost

        generations += 1

    return best_state, best_cost, generations, evaluations


# Algoritmo de Busca Aleatória
def randomSearch(state, doors, iterations):
    generations = 0
    evaluations = 0
    # Inicialize o melhor estado com o estado inicial
    for _ in range(iterations):
        neighbor = generateNeighbor(state, doors)
        evaluations += 1

        if neighbor is not None and neighbor.cost < state.cost:
            state = neighbor

        generations += 1

    return state, generations, evaluations

# Função para verificar se todas as casas podem ser alcançadas por todas as portas  
def can_reach_all_doors(mapa, house, doors):
    for door in doors:
        if not hasPath(mapa, house, door):
            return False
    return True

# Função para verificar se existe um caminho entre duas posições
def hasPath(mapa, start, end):
    visited = np.zeros(mapa.shape, dtype=bool)
    q = Queue()
    q.put(start)
    visited[start.row][start.col] = True

    dr = [-1, 1, 0, 0, -1, -1, 1, 1]
    dc = [0, 0, -1, 1, -1, 1, -1, 1]

    while not q.empty():
        curr = q.get()

        if curr == end:
            return True

        for i in range(8):
            newRow = curr.row + dr[i]
            newCol = curr.col + dc[i]

            if isValidPosition(mapa, newRow, newCol) and not visited[newRow][newCol] and mapa[newRow][newCol] != -1:
                q.put(Position(newRow, newCol))
                visited[newRow][newCol] = True

    return False

# Função para calcular o custo de um estado
def calculate_cost(mapa, doors, num_zonas_habitacionais=0, num_zonas_circulacao=0, num_zonas_inuteis=0):
    cost = 0
    for door in doors:
        connectedHouses = getConnectedHouses(mapa, door)
        for house in connectedHouses:
            dist = sqrt((house.row - door.row) ** 2 + (house.col - door.col) ** 2)
            cost += dist
    # Adicione o custo de cada zona
    for row in range(mapa.shape[0]):
        for col in range(mapa.shape[1]):
            if mapa[row][col] == 1 and not can_reach_all_doors(mapa, Position(row, col), doors):
                cost += 100

    return cost

# Função para implementar o algoritmo de Simulated Annealing
def Simulated_Annealing(mapa, doors, T, alpha):
    state = State(mapa, calculate_cost(mapa, doors))
    generations = 0
    evaluations = 0
    # Enquanto a temperatura for maior que 1e-6, continue a busca
    while T > 1e-6:
        newState = generateNeighbor(state, doors)
        evaluations += 1

        delta = newState.cost - state.cost

        if delta < 0:
            state = newState
        elif np.random.random() < exp(-delta / T):
            state = newState

        T *= alpha
        generations += 1

    return state, generations, evaluations

# Função para implementar o algoritmo genético
def genetic_algorithm(mapa, num_iterations):
    best_map = mapa.copy()
    best_cost = calculate_cost_violations(best_map)
    best_state = State(best_map, best_cost)  # Inicializa o melhor estado com o estado inicial
    evaluations = 1  # Inicializa o contador de avaliações
    generations = 0  # Inicializa o contador de gerações

    for _ in range(num_iterations):
        next_map = best_map.copy()
        random_move(next_map)
        next_cost = calculate_cost_violations(next_map)
        evaluations += 1  # Incrementa o contador de avaliações

        if next_cost < best_cost:
            best_map = next_map
            best_cost = next_cost
            best_state = State(next_map, next_cost, moves=best_state.moves + 1)  # Atualiza o número de movimentos


        # Verifica se uma nova geração foi alcançada
        if _ % len(mapa) == 0:  # A cada len(mapa) iterações
            generations += 1  # Incrementa o contador de gerações

    return best_state, generations, evaluations

# Função de avaliação modificada para incluir penalização por violações
def calculate_cost_violations(mapa):
    blocked_portals = 0
    inaccessible_zones = 0
    violations = 0
    for i in range(len(mapa)):
        for j in range(len(mapa[0])):
            if mapa[i][j] == 0:
                blocked_portals += len([nbr for nbr in get_neighbors(mapa, i, j) if mapa[nbr] == 1])
                if not has_access_to_all_doors(mapa, (i, j)):
                    inaccessible_zones += 1
                    violations += 100  # Penalização por violação
    return blocked_portals + 100 * inaccessible_zones + violations

# Função para verificar se todas as casas podem ser alcançadas por todas as portas
def has_access_to_all_doors(mapa, position):
    # Verifique se temos um caminho para todas as 4 portas a partir desta posição
    doors = [(0, 2), (4, 2), (2, 0), (2, 4)]  # posição das portas no mapa

    for door in doors:
        if not path_exists(mapa, position, door):
            return False  # Se não conseguirmos encontrar um caminho para uma porta, retorne False
    return True  # Se conseguimos encontrar um caminho para todas as portas, retorne True

# Função para encontrar todas as portas de saída no mapa
def find_exit_doors(mapa):
    exit_doors = []
    for i in range(len(mapa)):
        for j in range(len(mapa[0])):
            if mapa[i][j] == 1:
                if (i == 0 or i == len(mapa) - 1 or j == 0 or j == len(mapa[0]) - 1):
                    exit_doors.append((i, j))
    return exit_doors

# Função para implementar o algoritmo de Busca Aleatória
def random_move(mapa):
    # Procurar todas as zonas habitacionais e áreas livres usando np.where()
    zones = np.where(mapa == 1)
    zones = list(zip(zones[0], zones[1]))
    free_spaces = np.where(mapa == 0)
    free_spaces = list(zip(free_spaces[0], free_spaces[1]))

    # Embaralhar as listas para selecionar aleatoriamente
    random.shuffle(zones)
    random.shuffle(free_spaces)

    doors = [(0, 2), (4, 2), (2, 0), (2, 4)]  # posição das portas no mapa

    if zones and free_spaces:
        zone = zones[0]
        free_space = free_spaces[0]

        # Verifique se a zona não está se movendo para uma porta
        if free_space in doors:
            return mapa  # Não realiza movimento

        # Tenta mover a zona para a área livre
        mapa[zone[0]][zone[1]] = 0
        mapa[free_space[0]][free_space[1]] = 1

        # Verifica se o mapa ainda é válido após a mudança
        if has_access_to_all_doors(mapa, (free_space[0], free_space[1])):
            return mapa  # Realiza movimento
        else:
            # Desfaz a mudança se o mapa não for válido
            mapa[zone[0]][zone[1]] = 1
            mapa[free_space[0]][free_space[1]] = 0

    return mapa  # Não realiza movimento

# Função para plotar o mapa
def plot_map(mapa):
    cmap = plt.cm.get_cmap("Pastel1")
    colors = [cmap(0), cmap(1), cmap(2)]
    border_color = "#000000"
    # Crie uma figura e um eixo
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")
    # Desenhe o mapa
    for i in range(len(mapa)):
        for j in range(len(mapa[0])):
            if mapa[i][j] == 0:
                ax.add_patch(plt.Rectangle((j, len(mapa) - 1 - i), 1, 1, facecolor=colors[0], edgecolor=border_color))
            elif mapa[i][j] == 1:
                ax.add_patch(plt.Rectangle((j, len(mapa) - 1 - i), 1, 1, facecolor=colors[1], edgecolor=border_color))
            elif mapa[i][j] == -1:
                ax.add_patch(plt.Rectangle((j, len(mapa) - 1 - i), 1, 1, facecolor=colors[2], edgecolor=border_color))
    # Ajuste a escala do eixo
    ax.autoscale_view()
    fig.set_size_inches(8, 8)
    return fig, ax

# Mapas para testar o algoritmo
mapas = [
    np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, -1, -1, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 0, 0]
    ]),
 np.array([
        [0, 1, 1, 0, 0],
        [1, 1, -1, 1, 1],
        [0, -1, -1, -1, 0],
        [1, 1, -1, 1, 0],
        [0, 1, 0, 1, 1]
    ]),
np.array ([
    [0, 1, 1, 0, 0],
    [1, -1, 0, 1, 1],
    [1, -1, -1, -1, 0],
    [0, 1, 1, -1, 0],
    [0, 1, 0, 1, 1]
]),
np.array ([
    [1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 1, 1]
]),
np.array ([
    [1, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, -1, 1],
    [0, 1, 0, 0, 1, -1, 0],
    [1, -1, -1, -1, 1, -1, 1],
    [1, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 0]
]),
np.array ([
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0],
    [1, 0, -1, -1, -1, 0, 1],
    [0, 1, -1, -1, -1, 1, 0],
    [1, 1, -1, -1, -1, 0, 1],
    [0, 1, 1, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1, 0]
]),
np.array ([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]
]),
np.array ([
    [0, 1, 1, 1, 0, -1, 0, 0, 0],
    [1, 1, 0, 1, 1, -1, 1, 1, 1],
    [0, 1, -1, -1, 0, 0, 0, 0, 0],
    [1, 1, 0, -1, -1, -1, 0, 1, 0],
    [1, 1, 0, -1, 0, -1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, -1, -1, -1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0]
]),
np.array ([
    [0, 1, 1, 1, 0, 0, 0, 0, -1],
    [1, 1, -1, 1, 1, 1, 1, 1, -1],
    [0, 1, -1, 1, 0, 1, -1, 0, -1],
    [-1, -1, -1, 1, 0, 1, -1, 1, 0],
    [1, 1, 0, 1, 0, 1, -1, 1, 0],
    [1, 0, -1, -1, -1, -1, -1, 1, 0],
    [1, 0, 1, 0, 0, 0, -1, 1, -1],
    [0, 1, 0, 1, 0, 1, -1, 0, -1],
    [0, 0, 0, 1, 0, 0, 0, 0, -1]   
]),
np.array ([
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
])
]

# Função para gerar o link de download da tabela CSV 
def get_download_link():
    with open('tabela.csv', 'r') as file:
        csv_data = file.read()
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="tabela.csv">Download da tabela CSV</a>'
    return href
      
# Função principal para execução do código
def main():
    # Definindo o título do aplicativo
    st.title("Algoritmos de Busca - Problema da Cidade Muralhada Medieval")
    
    # Adicionando uma barra lateral para o aplicativo
    st.sidebar.title("EfolioB - IIA")
    st.sidebar.markdown("Ivo - 2100729")
    
    # Adicionando controles de usuário para o mapa e o algoritmo
    map_index = st.sidebar.selectbox("Escolha o mapa:", options=range(1, len(mapas) + 1), format_func=lambda x: f"Mapa {x}") - 1
    mapa = mapas[map_index]
    algorithm = st.sidebar.selectbox("Escolha o algoritmo:", 
                                     ("Simulated Annealing", "Hill Climbing", "Genetic Algorithm", "Busca Aleatória"))
    
    # Adicionando controles de usuário para o número de iterações
    iterations = st.sidebar.number_input('Número de Iterações', min_value=1, value=1000)
    
    # Adicionando controles de usuário para o fator de resfriamento do Simulated Annealing
    cooling_factor = st.sidebar.slider('Fator de Resfriamento', min_value=0.01, max_value=1.0, value=0.99)

    # Definindo as portas para cada mapa
    doors = [Position(0, 2), Position(2, 0), Position(4, 2), Position(2, 4)]
    
    # Definindo o estado inicial
    initState = State(mapa, 0, 0) 
    initial_map = mapa.copy()
    initial_cost = calculate_cost(initial_map, doors)
    initial_state = State(initial_map, initial_cost, 0)
    
    # Definindo os parâmetros para o Simulated Annealing
    T = 100.0
    alpha = cooling_factor 
    cost = 0  # Ou qualquer outro valor inicial desejado
    evaluations = 0
    generations = 0
    execution_time = 0
    costinitial = 0
    bestcost = 0
    num_map = int(map_index) + 1
   
    # Definir o tempo de início como 0
    start_time = 0
    
    # Definir os valores para cada mapa
    num_zonas_habitacionais = [13, 12, 11, 17, 20, 20, 34, 27, 30, 57]
    num_zonas_circulacao = [10, 8, 9, 8, 23, 20, 15, 42, 30, 24]
    num_zonas_inuteis = [2, 5, 5, 8, 6, 9, 0, 12, 21, 0]
    custos_referencia = [3, 4, 4, 4, 3, 2, 9, 11, 12, 13, 4]

    # Escolher o mapa desejado
    mapa_index = 0  # Índice do mapa que você deseja usar
    num_zonas_habitacionais_mapa = num_zonas_habitacionais[mapa_index]
    num_zonas_circulacao_mapa = num_zonas_circulacao[mapa_index]
    num_zonas_inuteis_mapa = num_zonas_inuteis[mapa_index]
    custo_referencia_mapa = custos_referencia[mapa_index]
        
    # Calcular o custo inicial considerando os valores específicos do mapa
    initial_cost = calculate_cost(initial_map, doors, num_zonas_habitacionais_mapa, num_zonas_circulacao_mapa, num_zonas_inuteis_mapa)
    
    # Exibir o mapa inicial
    st.write("Mapa Inicial:")
    st.write(np.array(mapa))
    fig, ax = plot_map(mapa)
    st.pyplot(fig)
    
    # Botão para executar o algoritmo
    if st.button("Encontrar solução"):
        
        # Registre o tempo inicial
        start_time = time.time()# Quando o usuário clicar no botão, execute o algoritmo selecionado
        initial_state = State(mapa, calculate_cost(mapa, doors))
        
        # Verifique qual algoritmo foi selecionado e execute-o
        if algorithm == "Simulated Annealing":
            solution, generations, evaluations = Simulated_Annealing(mapa, doors, T, alpha)
            st.write("Custo inicial:", int(calculate_cost(mapa, doors)))
            costinitial = calculate_cost(mapa, doors)
            st.markdown(f"**Custo mínimo encontrado:** {int(solution.cost)}")
            bestcost = solution.cost
            st.markdown(f"**Número mínimo de movimentos:** {solution.moves}") 

            # Mostrar o mapa final
            st.write("Mapa Final:")
            st.write(np.array(solution.mapa))
            fig, ax = plot_map(solution.mapa)
            st.pyplot(fig)
            # Comparar o custo mínimo encontrado com o custo de referência do mapa
            if solution.cost <= custo_referencia_mapa:
                st.markdown("**Custo mínimo encontrado atende ao requisito de referência.**")
            else:
                st.markdown("**Custo mínimo encontrado não atende ao requisito de referência.**")
            
        elif algorithm == "Hill Climbing":
                solution, cost, generations, evaluations = hill_climbing(State(mapa, calculate_cost(mapa, doors)), doors, iterations)
                st.write("Custo inicial:", int(calculate_cost(mapa, doors)))
                costinitial = calculate_cost(mapa, doors)
                st.write("Melhor custo encontrado:", int(cost))
                bestcost = cost
                st.markdown(f"**Número mínimo de movimentos:** {solution.moves}")

                st.write("Mapa Final:")
                st.write(solution.mapa)
                fig, ax = plot_map(solution.mapa)
                st.pyplot(fig)
                  # Comparar o custo mínimo encontrado com o custo de referência do mapa
                if solution.cost <= custo_referencia_mapa:
                    st.markdown("**Custo mínimo encontrado atende ao requisito de referência.**")
                else:
                    st.markdown("**Custo mínimo encontrado não atende ao requisito de referência.**")
                    
        if algorithm == "Genetic Algorithm":
                solution, generations, evaluations = genetic_algorithm(mapa, iterations)
                st.write("Custo inicial:", int(calculate_cost(mapa, doors)))
                costinitial = calculate_cost(mapa, doors)
                st.write("Melhor custo encontrado:", int(solution.cost))
                bestcost = solution.cost
                # st.markdown(f"**Número de avaliações:** {evaluations}")
                # st.markdown(f"**Número de gerações:** {generations}")
                
                # Acessar o atributo 'cost' do objeto 'State' dentro do 'solution'
                if isinstance(solution, np.ndarray) and len(solution) > 0:
                    if cost <= custo_referencia_mapa:
                        st.markdown("**Custo mínimo encontrado atende ao requisito de referência.**")
                    else:
                        st.markdown("**Custo mínimo encontrado não atende ao requisito de referência.**")
                        
                # Acessar o atributo 'mapa' do objeto 'State'
                if hasattr(solution, 'mapa'):
                    st.write("Mapa Final:")
                    st.write(solution.mapa)
                    fig, ax = plot_map(solution.mapa)
                    st.pyplot(fig)
                
                # Acessar o atributo 'moves' do objeto 'State'
                if hasattr(solution, 'moves'):
                    st.markdown(f"**Número mínimo de movimentos:** {solution.moves}")

                  # Comparar o custo mínimo encontrado com o custo de referência do mapa
                if solution.cost <= custo_referencia_mapa:
                    st.markdown("**Custo mínimo encontrado atende ao requisito de referência.**")
                else:
                    st.markdown("**Custo mínimo encontrado não atende ao requisito de referência.**")
            
        elif algorithm == "Busca Aleatória":
                solution, generations, evaluations = randomSearch(State(mapa, calculate_cost(mapa, doors)), doors, iterations)
                st.write("Custo inicial:", int(calculate_cost(mapa, doors)))
                costinitial = calculate_cost(mapa, doors)
                st.markdown(f"**Custo mínimo encontrado:** {int(solution.cost)}")
                bestcost = solution.cost
                st.markdown(f"**Número mínimo de movimentos:** {solution.moves}")
                st.write("Mapa Final:")
                st.write(np.array(solution.mapa))
                fig, ax = plot_map(solution.mapa)
                st.pyplot(fig)

                # Comparar o custo mínimo encontrado com o custo de referência do mapa
                if solution.cost <= custo_referencia_mapa:
                    st.markdown("**Custo mínimo encontrado atende ao requisito de referência.**")
                else:
                    st.markdown("**Custo mínimo encontrado não atende ao requisito de referência.**")
                
    # Mostrar o tempo de execução, número de gerações e número de avaliações
    if start_time > 0:
            # Calcula o tempo de execução apenas se o botão foi clicado
            # Calcula o tempo de execução em milissegundos
            execution_time = (time.time() - start_time) * 1000
            st.write("Tempo de execução:", execution_time, "milissegundos")

            # Mostrar o número de gerações e avaliações
            st.write("Gerações:", generations)
            st.write("Avaliações:", evaluations)

    # Adicionar botão na barra lateral
    if st.sidebar.button("Fazer download da tabela"):
         st.sidebar.markdown(get_download_link(), unsafe_allow_html=True)

    # Definir os valores para cada mapa
    mapa_selecionado = {
    'Instância': [num_map],
    'Algoritmo': [algorithm],
    'Avaliações': [evaluations],
    'Gerações': [generations],
    'Custo': [costinitial],
    'Tempo (msec)': [execution_time],
    'Melhor': [bestcost]
}

    # Verificar se o arquivo "tabela.csv" já existe
    arquivo = 'tabela.csv'
    existe_arquivo = os.path.isfile(arquivo)
    
    # Se o arquivo não existir, criar um novo arquivo CSV
    if not existe_arquivo:
        df = pd.DataFrame(mapa_selecionado)
    else:
        # Ler arquivo existente
        df = pd.read_csv(arquivo)

    # Adicionar dados do mapa selecionado à tabela existente
    df_mapa_selecionado = pd.DataFrame(mapa_selecionado)
    df = pd.concat([df, df_mapa_selecionado], ignore_index=True)
    
    # Salvar tabela em um arquivo CSV sem apagar os dados existentes
    df.to_csv(arquivo, index=False)
    
# Executar o código
if __name__ == "__main__":
    main()
