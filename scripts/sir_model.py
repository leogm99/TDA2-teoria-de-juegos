import copy

import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum
from random import random
import seaborn as sns
import pandas as pd
import multiprocessing as mp

import numpy as np


class NodeState(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2


def get_random_node(g: nx.Graph):
    from random import choice
    return choice(list(g.nodes))


def get_highest_degree_node(g: nx.Graph):
    return max(g.degree, key=lambda x: x[1])[0]


def sir(g: nx.Graph, source_node, beta, delta, epochs, epidemic_threshold=0.3):
    """

    :param g: Source graph to calculate SIR model on.
    :param source_node: Callable which yields the first infected node from g.
    :param beta: Probability that a susceptible individual gets infected
    :param delta: Probability at any given time that the earliest current infected node gets recovered
    :param epochs: How many discrete time states, at most, to run the simulation.
    :param epidemic_threshold: Percentage of population requiered to qualify as epidemic
    """

    def is_node_in_state(node, state):
        return g.nodes[node]['node_state'] == state

    def set_node_state(node, new_state):
        g.nodes[node]['node_state'] = new_state

    def get_quantity_in_state(state):
        return len(list(filter(lambda x: is_node_in_state(x, state), g.nodes)))

    nx.set_node_attributes(g, NodeState.SUSCEPTIBLE, 'node_state')

    # source node is the first to get infected
    src = source_node(g)
    set_node_state(src, NodeState.INFECTED)

    infected = [src]
    to_infect = []
    starting = True
    data_out = []
    i = 0
    epidemic_ocurred = False
    for i in range(epochs):
        q_susceptible = get_quantity_in_state(NodeState.SUSCEPTIBLE)
        q_recovered = get_quantity_in_state(NodeState.RECOVERED)
        q_infected = get_quantity_in_state(NodeState.INFECTED)
        if q_infected >= epidemic_threshold * len(g.nodes):
            epidemic_ocurred = True
        data_out.append((q_susceptible, q_recovered, q_infected))

        for j in infected:
            got_recovered = random() < delta
            if got_recovered and not starting:
                set_node_state(j, NodeState.RECOVERED)
        infected = list(filter(lambda x: is_node_in_state(x, NodeState.INFECTED), infected))
        starting = False
        # epidemic ended
        if not infected:
            break

        to_infect.clear()
        for j in infected:
            for n in g.neighbors(j):
                if is_node_in_state(n, NodeState.SUSCEPTIBLE):
                    got_infected = random() < beta
                    if got_infected:
                        to_infect.append(n)

        infected.extend(to_infect)
        for n in to_infect:
            set_node_state(n, NodeState.INFECTED)

    # pprint(nx.get_node_attributes(g, 'node_state'))
    susceptible_data = np.array(list(map(lambda x: x[0], data_out)))
    recovered_data = np.array(list(map(lambda x: x[1], data_out)))
    infected_data = np.array(list(map(lambda x: x[2], data_out)))
    return susceptible_data, recovered_data, infected_data, i, epidemic_ocurred


def par_sample(*args):
    data = {}
    beta_values = args[0]
    print(beta_values)
    delta_values = args[1]
    g = args[2]
    src_node_call = args[3]
    epochs = args[4]

    iters = len(beta_values) * len(delta_values)
    curr_iter = 1
    for b in beta_values:
        for d in delta_values:
            print(f"[Process: {mp.current_process()}] Sampling iteration {curr_iter} of {iters}")
            *_, epidemic_ocurred = sir(g, src_node_call, b, d, epochs)
            data[(b, d)] = epidemic_ocurred
            if epidemic_ocurred:
                print(f"Epidemic ocurred")
            curr_iter += 1

    return data


def sample(g: nx.Graph, src_node_callable, beta, max_beta, delta, max_delta, epochs):
    betas = np.linspace(beta, max_beta, 100)
    deltas = np.linspace(delta, max_delta, 100)

    beta_values_splitted = np.array_split(betas, mp.cpu_count())

    with mp.Pool(mp.cpu_count()) as p:
        new_g = copy.deepcopy(g)
        all_data = p.starmap(par_sample, [(beta_values_splitted[0], deltas, new_g, src_node_callable, epochs),
                                          (beta_values_splitted[1], deltas, new_g, src_node_callable, epochs),
                                          (beta_values_splitted[2], deltas, new_g, src_node_callable, epochs),
                                          (beta_values_splitted[3], deltas, new_g, src_node_callable, epochs)])
        return {k: v for d in all_data for k, v in d.items()}


def plot_data(susceptible, recovered, infected, epochs_lasted, title=""):
    x_axis = np.array(range(epochs_lasted + 1))
    df = pd.DataFrame({'epochs': x_axis, 'susceptible': susceptible, 'recovered': recovered, 'infected': infected},
                      index=list(range(epochs_lasted + 1)))
    ax = sns.scatterplot(data=df, x='epochs', y='susceptible', )
    ax.set(ylabel="Individuals", title=title)
    # TODO: remove hardcoding
    ax.axhline(.3 * 500, ls='--')
    sns.scatterplot(data=df, x='epochs', y='recovered')
    sns.scatterplot(data=df, x='epochs', y='infected')
    plt.show()


def main():
    net = nx.erdos_renyi_graph(n=500, p=0.1)
    net2 = nx.powerlaw_cluster_graph(n=500, m=26, p=0.0)
    epochs = 500
    data = sample(net, get_random_node, beta=0.01, max_beta=0.1, delta=0.1, max_delta=0.5, epochs=epochs)
    print("done")
    # sample(net, 100, get_random_node, 0.01, 0.1, epochs)
    # random source node
    """
    susceptible, recovered, infected, lasted, _ = sir(net, get_random_node, 0.01, 0.1, epochs)
    plot_data(susceptible, recovered, infected, lasted, "Erdos-Renyi epidemic")
    susceptible, recovered, infected, lasted, _ = sir(net2, get_random_node, 0.01, 0.1, epochs)
    plt.cla()
    plot_data(susceptible, recovered, infected, lasted, "Powerlaw epidemic")
    """


if __name__ == '__main__':
    main()
