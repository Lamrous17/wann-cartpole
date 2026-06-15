import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display
import time
import pickle
from copy import deepcopy
import os
import networkx as nx

import warnings
warnings.filterwarnings('ignore')

# 1. Класс WANNGenome
class WANNGenome:
    """Геном: узлы и связи (веса фиксированы)"""
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nodes = set(range(num_inputs + num_outputs))
        self.connections = {}
        for i in range(num_inputs):
            for o in range(num_inputs, num_inputs + num_outputs):
                self.connections[(i, o)] = True
        self.next_node_id = num_inputs + num_outputs
        self.fitness = None

    def add_node(self):
        """Добавить скрытый нейрон"""
        if not self.connections:
            return
        conn = random.choice(list(self.connections.keys()))
        if not self.connections[conn]:
            return
        self.connections[conn] = False   # отключаем старую связь
        from_node, to_node = conn
        new_node = self.next_node_id
        self.next_node_id += 1
        self.nodes.add(new_node)
        self.connections[(from_node, new_node)] = True
        self.connections[(new_node, to_node)] = True

    def add_connection(self):
        """Добавить связь"""
        possible = []
        nodes_list = list(self.nodes)
        for i in nodes_list:
            for j in nodes_list:
                if i == j:
                    continue
                if i < self.num_inputs and j < self.num_inputs: continue
                if i >= self.num_inputs and j < self.num_inputs:
                    continue
                if (i, j) not in self.connections:
                    if not self._is_reachable(j, i):
                        possible.append((i, j))
        if possible:
            from_node, to_node = random.choice(possible)
            self.connections[(from_node, to_node)] = True

    def remove_connection(self):
        """Удаляем случайную активную связь"""
        enabled = [c for c, en in self.connections.items() if en]
        if enabled:
            del self.connections[random.choice(enabled)]

    def remove_node(self):
        """Удаляем скрытый узел"""
        hidden = [n for n in self.nodes if n >= self.num_inputs + self.num_outputs]
        if hidden:
            node = random.choice(hidden)
            self.nodes.remove(node)
            self.connections = {c: en for c, en in self.connections.items() if node not in c}

    def _is_reachable(self, start, target, visited=None):
        """Проверка на цикл"""
        if visited is None:
            visited = set()
        if start == target:
            return True
        visited.add(start)
        for (f, t), en in self.connections.items():
            if en and f == start and t not in visited:
                if self._is_reachable(t, target, visited):
                    return True
        return False

    def forward(self, inputs):
        values = {i: 0.0 for i in self.nodes}
        for i, val in enumerate(inputs):
            if i in values:
                values[i] = val
        for _ in range(len(self.nodes)):
            for (f, t), enabled in self.connections.items():
                if enabled:
                    values[t] += values[f]
                    if t >= self.num_inputs:
                        values[t] = np.tanh(values[t])
        outputs = [values[self.num_inputs + i] for i in range(self.num_outputs)]
        return np.argmax(outputs)

    def mutate(self, mutation_probs):
        r = random.random()
        if r < mutation_probs['add_node']:
            self.add_node()
        elif r < mutation_probs['add_node'] + mutation_probs['add_conn']:
            self.add_connection()
        elif r < mutation_probs['add_node'] + mutation_probs['add_conn'] + mutation_probs['remove_conn']:
            self.remove_connection()
        elif r < mutation_probs['add_node'] + mutation_probs['add_conn'] + mutation_probs['remove_conn'] + mutation_probs['remove_node']:
            self.remove_node()

    def clone(self):
        return deepcopy(self)


# 2. Вспомогательные функции
def eval_genome(genome, env, max_steps=500):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated) and step < max_steps:
        action = genome.forward(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
    return total_reward


def tournament_selection(population, scores, tournament_size=3):
    best_idx = random.randrange(len(population))
    for _ in range(tournament_size - 1):
        idx = random.randrange(len(population))
        if scores[idx] > scores[best_idx]:
            best_idx = idx
    return population[best_idx]


def evolve_wann(num_inputs, num_outputs, env, pop_size=100, generations=50,
                mutation_probs=None, elite_ratio=0.1, tournament_size=3):
    if mutation_probs is None:
        mutation_probs = {'add_node': 0.1, 'add_conn': 0.2, 'remove_conn': 0.1, 'remove_node': 0.05}

    population = [WANNGenome(num_inputs, num_outputs) for _ in range(pop_size)]

    best_fitness_history = []
    avg_fitness_history = []
    complexity_history = []
    # Новые метрики
    nodes_history = []
    params_history = []
    best_networks = {}

    best_genome = None

    for gen in range(generations):
        fitnesses = []
        for g in population:
            fitness = eval_genome(g, env)
            g.fitness = fitness
            fitnesses.append(fitness)

        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        avg_fitness = np.mean(fitnesses)

        # Обновление лучшего генома
        if best_genome is None or best_fitness > max(best_fitness_history):
            best_genome = population[best_idx].clone()

        # Сохранение лучших сетей на разных этапах
        if gen == 0:
            best_networks["early"] = best_genome.clone()
        if gen == generations // 2:
            best_networks["middle"] = best_genome.clone()
        if gen == generations - 1:
            best_networks["late"] = best_genome.clone()

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        # Среднее число связей
        avg_conn = np.mean([len(g.connections) for g in population])
        complexity_history.append(avg_conn)

        # Среднее число нейронов
        avg_nodes = np.mean([len(g.nodes) for g in population])
        nodes_history.append(avg_nodes)

        # Среднее число параметров (активных связей)
        avg_params = np.mean([sum(g.connections.values()) for g in population])
        params_history.append(avg_params)

        # Элитизм и турнирный отбор
        elite_count = int(pop_size * elite_ratio)
        sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        new_population = [g.clone() for g in sorted_pop[:elite_count]]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            child = parent1.clone() if parent1.fitness >= parent2.fitness else parent2.clone()
            child.mutate(mutation_probs)
            new_population.append(child)

        population = new_population

        # Логирование каждые 5 поколений
        if gen % 5 == 0 or gen == generations - 1:
            print(f"Поколение {gen}: best={best_fitness:.2f}, avg={avg_fitness:.2f}, connections={avg_conn:.1f}")

            # Живой график процесса эволюции
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(gen + 1), best_fitness_history, 'b-', label='Лучший')
            plt.plot(range(gen + 1), avg_fitness_history, 'r--', label='Средний')
            plt.xlabel('Поколение')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(gen + 1), complexity_history, 'g-', label='Среднее число связей')
            plt.xlabel('Поколение')
            plt.ylabel('Сложность')
            plt.legend()
            plt.grid(True)

            plt.suptitle(f'Эволюция WANN (поколение {gen})')
            plt.tight_layout()
            plt.savefig('evolution_plot.png', dpi=150)
            display.clear_output(wait=False)
            plt.show()
            plt.close()

    # Финальный анализ сложности
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(best_fitness_history, label="Лучший")
    plt.plot(avg_fitness_history, label="Средний")
    plt.xlabel("Поколение")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(complexity_history)
    plt.title("Среднее число связей")
    plt.xlabel("Поколение")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(nodes_history)
    plt.title("Среднее число нейронов")
    plt.xlabel("Поколение")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(params_history)
    plt.title("Среднее число параметров")
    plt.xlabel("Поколение")
    plt.grid()

    plt.tight_layout()
    plt.savefig("complexity_analysis.png")
    plt.show()

    return (best_genome,
            best_fitness_history,
            avg_fitness_history,
            complexity_history,
            nodes_history,
            params_history,
            best_networks)


def draw_network(genome, title):

    G = nx.DiGraph()

    input_nodes = list(range(genome.num_inputs))
    output_nodes = list(range(genome.num_inputs,
                              genome.num_inputs + genome.num_outputs))
    hidden_nodes = sorted(
        list(genome.nodes - set(input_nodes) - set(output_nodes))
    )

    for n in genome.nodes:
        G.add_node(n)

    for (u, v), enabled in genome.connections.items():
        if enabled:
            G.add_edge(u, v)

    pos = {}

    # входы
    for i, node in enumerate(input_nodes):
        pos[node] = (0, -i)

    # скрытые
    for i, node in enumerate(hidden_nodes):
        pos[node] = (1, -i)

    # выходы
    for i, node in enumerate(output_nodes):
        pos[node] = (2, -i)

    plt.figure(figsize=(8,6))

    nx.draw_networkx_nodes(G, pos,
                           nodelist=input_nodes,
                           node_color="lime",
                           node_size=600,
                           label="Input")

    nx.draw_networkx_nodes(G, pos,
                           nodelist=hidden_nodes,
                           node_color="skyblue",
                           node_size=600,
                           label="Hidden")

    nx.draw_networkx_nodes(G, pos,
                           nodelist=output_nodes,
                           node_color="tomato",
                           node_size=600,
                           label="Output")

    nx.draw_networkx_edges(G, pos,
                           arrows=True,
                           arrowsize=20)

    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()


# 3. Запуск эволюции на CartPole-v1
env = gym.make('CartPole-v1')
num_inputs = env.observation_space.shape[0]   # 4
num_outputs = env.action_space.n              # 2
print(f"Среда: CartPole-v1, входов={num_inputs}, выходов={num_outputs}")

(best,
 best_hist,
 avg_hist,
 comp_hist,
 node_hist,
 param_hist,
 best_networks) = evolve_wann(
    num_inputs, num_outputs, env,
    pop_size=80,
    generations=40,
    mutation_probs={'add_node': 0.08, 'add_conn': 0.2, 'remove_conn': 0.1, 'remove_node': 0.04},
    elite_ratio=0.1,
    tournament_size=3
)
print(f"\nЛучшая приспособленность: {max(best_hist):.2f}")

# 4. Baseline: случайный агент
def random_agent(env, episodes=10):
    total = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        rew = 0
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            rew += r
            done = terminated or truncated
        total += rew
    return total / episodes

random_score = random_agent(env)
print(f"Случайный агент (среднее): {random_score:.2f}")
print(f"WANN агент: {max(best_hist):.2f}")

# Сравнение по поколениям
baseline = [random_score] * len(best_hist)
plt.figure(figsize=(8, 5))
plt.plot(best_hist, label="WANN")
plt.plot(baseline, '--', label="Random")
plt.xlabel("Поколение")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.savefig("baseline_compare.png")
plt.show()

# Столбчатая диаграмма итогового сравнения
plt.figure(figsize=(6, 4))
plt.bar(['Случайный', 'WANN'], [random_score, max(best_hist)], color=['gray', 'green'])
plt.ylabel('Средняя награда')
plt.title('Сравнение: WANN vs Random')
plt.savefig('comparison.png')
plt.show()
plt.close()

# 5. Демонстрация лучшего агента
def render_agent(genome, episodes=2):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    for ep in range(episodes):
        obs, _ = env.reset()
        total = 0
        done = False
        step = 0
        while not done:
            action = genome.forward(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
            step += 1

            # Сохранение скриншотов поведения
            if step in [10, 50, 100]:
                plt.clf()
                plt.imshow(env.render())
                plt.axis('off')
                plt.savefig(f"agent_step_{step}.png")
                plt.close()

            # Отображение текущего кадра
            plt.clf()
            plt.imshow(env.render())
            plt.axis('off')
            display.clear_output(wait=True)
            display.display(plt.gcf())
            time.sleep(0.02)

        print(f"Эпизод {ep+1}: награда = {total:.2f}")
    env.close()
    plt.close()

print("\nДемонстрация лучшего агента WANN на CartPole:")
render_agent(best, episodes=2)

# Визуализация структуры сети
draw_network(best_networks["early"], "Ранняя сеть")
draw_network(best_networks["middle"], "Средняя сеть")
draw_network(best_networks["late"], "Финальная сеть")

# Сохранение модели
with open('wann_best_cartpole.pkl', 'wb') as f:
    pickle.dump(best, f)
print("\n💾 Модель сохранена как 'wann_best_cartpole.pkl'")

# 6. Гиперпараметрические эксперименты
print("Гиперпараметрические эксперименты")

# Эксперимент 1: размер популяции
pop_sizes = [40, 80, 120]
results_pop = []
for ps in pop_sizes:
    print(f"\nРазмер популяции = {ps}")
    _, best_h, _, _, _, _, _ = evolve_wann(num_inputs, num_outputs, env, pop_size=ps, generations=25,
                                           mutation_probs={'add_node':0.08,'add_conn':0.2,'remove_conn':0.1,'remove_node':0.04})
    results_pop.append(max(best_h))

plt.figure(figsize=(6, 4))
plt.bar([str(p) for p in pop_sizes], results_pop, color='orange')
plt.xlabel('Размер популяции')
plt.ylabel('Лучший fitness')
plt.title('Гиперпараметр: размер популяции')
plt.savefig('hyperpop.png')
plt.show()
plt.close()

# Эксперимент 2: вероятность добавления связи
conn_probs = [0.1, 0.3, 0.5]
results_conn = []
for cp in conn_probs:
    print(f"\nВероятность добавления связи = {cp}")
    mp = {'add_node':0.08,'add_conn':cp,'remove_conn':0.1,'remove_node':0.04}
    _, best_h, _, _, _, _, _ = evolve_wann(num_inputs, num_outputs, env, pop_size=60, generations=25, mutation_probs=mp)
    results_conn.append(max(best_h))

plt.figure(figsize=(6, 4))
plt.bar([str(c) for c in conn_probs], results_conn, color='purple')
plt.xlabel('Вероятность add_connection')
plt.ylabel('Лучший fitness')
plt.title('Гиперпараметр: мутация добавления связи')
plt.savefig('hyperconn.png')
plt.show()
plt.close()

# 7. Абляция (отключение добавления узлов)
print("\nАбляция: отключение добавления узлов")
mp_no_node = {'add_node':0.0, 'add_conn':0.2, 'remove_conn':0.1, 'remove_node':0.0}
_, best_h_no_node, _, _, _, _, _ = evolve_wann(num_inputs, num_outputs, env, pop_size=60, generations=25, mutation_probs=mp_no_node)
print(f"Без добавления узлов, лучший fitness: {max(best_h_no_node):.2f}")
print(f"С полным набором мутаций (из основного запуска): {max(best_hist):.2f}")

# 8. Проверка устойчивости (шум в наблюдениях)
print("Устойчивость к шуму")
def eval_genome_noisy(genome, env, noise_std=0.1, max_steps=500):
    obs, _ = env.reset()
    total = 0
    done = False
    step = 0
    while not done and step < max_steps:
        noisy_obs = obs + np.random.normal(0, noise_std, size=obs.shape)
        action = genome.forward(noisy_obs)
        obs, r, terminated, truncated, _ = env.step(action)
        total += r
        done = terminated or truncated
        step += 1
    return total

noisy_score = np.mean([eval_genome_noisy(best, env, noise_std=0.1) for _ in range(10)])
clean_score = np.mean([eval_genome(best, env) for _ in range(10)])
print(f"Чистая среда: {clean_score:.2f}")
print(f"С шумом (σ=0.1): {noisy_score:.2f}")

# 9. Анализ неудачных режимов
print("\nСлишком маленькая популяция (pop=10)")
_, bad_hist, _, _, _, _, _ = evolve_wann(num_inputs, num_outputs, env, pop_size=10, generations=30,
                                         mutation_probs={'add_node':0.08,'add_conn':0.2,'remove_conn':0.1,'remove_node':0.04})
print(f"Макс fitness при pop=10: {max(bad_hist):.2f} (против {max(best_hist):.2f} при pop=80)")

print("\nВысокая вероятность удаления связей (0.5)")
mp_high_rem = {'add_node':0.08, 'add_conn':0.2, 'remove_conn':0.5, 'remove_node':0.1}
_, bad_hist2, _, _, _, _, _ = evolve_wann(num_inputs, num_outputs, env, pop_size=60, generations=30, mutation_probs=mp_high_rem)
print(f"Высокая вероятность удаления связей, лучший fitness: {max(bad_hist2):.2f}")

def draw_network(genome, title):

    G = nx.DiGraph()

    input_nodes = list(range(genome.num_inputs))
    output_nodes = list(range(genome.num_inputs,
                              genome.num_inputs + genome.num_outputs))
    hidden_nodes = sorted(
        list(genome.nodes - set(input_nodes) - set(output_nodes))
    )

    for n in genome.nodes:
        G.add_node(n)

    for (u, v), enabled in genome.connections.items():
        if enabled:
            G.add_edge(u, v)

    pos = {}

    # входы
    for i, node in enumerate(input_nodes):
        pos[node] = (0, -i)

    # скрытые
    for i, node in enumerate(hidden_nodes):
        pos[node] = (1, -i)

    # выходы
    for i, node in enumerate(output_nodes):
        pos[node] = (2, -i)

    plt.figure(figsize=(8,6))

    nx.draw_networkx_nodes(G, pos,
                           nodelist=input_nodes,
                           node_color="lime",
                           node_size=600,
                           label="Input")

    nx.draw_networkx_nodes(G, pos,
                           nodelist=hidden_nodes,
                           node_color="skyblue",
                           node_size=600,
                           label="Hidden")

    nx.draw_networkx_nodes(G, pos,
                           nodelist=output_nodes,
                           node_color="tomato",
                           node_size=600,
                           label="Output")

    nx.draw_networkx_edges(G, pos,
                           arrows=True,
                           arrowsize=20)

    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()

# Вывод всех результатов
print("РЕЗУЛЬТАТЫ ОСНОВНОГО ЗАПУСКА")
print(f"Лучший fitness (основной запуск): {max(best_hist):.2f}")
print(f"Средний fitness в последнем поколении: {avg_hist[-1]:.2f}")
print(f"Среднее число связей в последнем поколении: {comp_hist[-1]:.2f}")

print("БЕЙСЛАЙН (случайный агент)")
print(f"Случайный агент: {random_score:.2f}")

print("ГИПЕРПАРАМЕТРЫ (результаты из экспериментов)")
print(f"Размер популяции 40 - лучший fitness: {results_pop[0] if 'results_pop' in dir() else 'не сохранён'}")
print(f"Размер популяции 80 - лучший fitness: {results_pop[1] if 'results_pop' in dir() else 'не сохранён'}")
print(f"Размер популяции 120 - лучший fitness: {results_pop[2] if 'results_pop' in dir() else 'не сохранён'}")
print(f"Вероятность add_conn 0.1 - {results_conn[0] if 'results_conn' in dir() else 'не сохранён'}")
print(f"Вероятность add_conn 0.3 - {results_conn[1] if 'results_conn' in dir() else 'не сохранён'}")
print(f"Вероятность add_conn 0.5 - {results_conn[2] if 'results_conn' in dir() else 'не сохранён'}")

print("АБЛЯЦИЯ (отключение добавления узлов)")
if 'best_h_no_node' in dir():
    print(f"Без добавления узлов: {max(best_h_no_node):.2f}")
else:
    print("Переменная best_h_no_node не найдена.")

print("УСТОЙЧИВОСТЬ К ШУМУ")
if 'clean_score' in dir():
    print(f"Чистая среда: {clean_score:.2f}")
    print(f"С шумом (σ=0.1): {noisy_score:.2f}")
else:
    print("Переменные clean_score/noisy_score не найдены")

print("НЕУДАЧНЫЕ РЕЖИМЫ")
if 'bad_hist' in dir():
    print(f"Популяция 10: {max(bad_hist):.2f}")
else:
    print("bad_hist не найден.")
if 'bad_hist2' in dir():
    print(f"Высокое удаление связей (0.5): {max(bad_hist2):.2f}")
else:
    print("bad_hist2 не найден.")
