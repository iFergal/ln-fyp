import networkx as nx
import matplotlib.pyplot as plt
from math import inf

"""
Units of transfer are presumed to be satoshi (0.00000001 BTC) - this is the smallest unit
available on BTC - in reality, the LN supports millisatoshi for fee rounding purposes.
-- Hence, fees are allowed be in the order of 0.0001 sat.

"""

class Node:
    """ Singular lightning node. """

    def __init__(self, id, fee_rate, onchain_amnt):
        self._id = id
        self._fee_rate = fee_rate  # [base, multiplier] - For now, presume fee rate does not change
        self._onchain_amnt = onchain_amnt

    def get_fees(self):
        return self._fee_rate

    def update_chan(self, G, dest, amnt):
        """Update a channel's balance by making a payment from src to dest.

        We assume both parties sign the update automatically for simulation purposes.
        Return True if successful, False otherwise.
        """
        if G.has_edge(self, dest):
            if G[self][dest]["equity_a"][0] == self:  # Determine direction
                side, other_side = "equity_a", "equity_b"
            else:
                side, other_side = "equity_b", "equity_a"

            # Assume: amnt > 0
            if G[self][dest][side][1] >= amnt:
                G[self][dest][side][1] -= amnt
                G[self][dest][other_side][1] += amnt
                return True
            else:
                print("Error: equity between %s and %s not available for transfer." % (self, dest))
        else:
            print("Error: no direct payment channel between %s and %s." % (self, dest))
        return False

    def make_payment(self, G, dest, amnt):
        """Make a regular payment from this node to destination node of amnt.

        Returns True if successful, False otherwise.
        """

        # Reduce graph to edges with enough equity - this is very costly - fix.
        searchable = nx.Graph(((src, tar, attr) for src, tar, attr in G.edges(data=True) \
                                if attr["equity"] > amnt))

        if self in searchable and dest in searchable and nx.has_path(searchable, self, dest):
            path = nx.shortest_path(searchable, self, dest, weight="equity")
            send_amnts = self._calc_with_path_fees(path, amnt)

            if len(path) > 20:
                print("Error: path exceeds max-hop distance.")
                return False

            for i in range(len(path)-1):
                hop = path[i].update_chan(G, path[i+1], send_amnts[i])
                if hop:
                    print("Sent %f from %s to %s." % (send_amnts[i], path[i], path[i+1]))
                else:
                    print("Payment failed.")
                    return False
        else:
            print("No route available.")
            return False
        return True

    def _calc_with_path_fees(self, path, amnt):
        """ Calculate the compound path fees required for a given path.

        Note: compounding as amnt of equity moving per node is different!
        """
        hop_amnts = [amnt]
        for node in path[1:-1][::-1]:  # Intermediate nodes, reversed
            fee_this_hop = node.get_fees()[0] + node.get_fees()[1] * hop_amnts[-1]
            hop_amnts.append(fee_this_hop + hop_amnts[-1])
        return hop_amnts[::-1]


    def __str__(self):
        return "Node %d" % self._id

@nx.utils.decorators.py_random_state(3)
def generate_wattz_strogatz(n, k, p, seed=None):
    """ As taken from NetworkX random_graph src code and modified. """
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.Graph()
    nodes = [Node(i, [1, 0.05], 500) for i in range(n)]

    # connect each node to k/2 neighbours
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        pairs = list(zip(nodes, targets))
        for pair in pairs:
            G.add_edge(pair[0], pair[1], equity=30, equity_a=[pair[0], 10], equity_b=[pair[1], 20])

    # rewire edges from each node
    # loop over all nodes in order (label) and neighbours in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is neighbours
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w, equity=30, equity_a=[u, 10], equity_b=[v, 20])
    return G

if __name__ == "__main__":
    G = generate_wattz_strogatz(100, 6, 0.3)
    nodes = list(G)
    a, b = nodes[0], nodes[10]
    # first = next(iter(G[a]))
    a.make_payment(G, b, 5)
    # nx.draw(G)
    # plt.show()
