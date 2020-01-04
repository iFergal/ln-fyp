import networkx as nx
from networkx.algorithms.shortest_paths.weighted import dijkstra_path as dijkstra_path
from networkx.generators.random_graphs import _random_subset
import matplotlib.pyplot as plt
from math import inf, floor
import time

"""
Units of transfer are presumed to be satoshi (0.00000001 BTC) - this is the smallest unit
available on BTC - in reality, the LN supports millisatoshi for fee rounding purposes.
-- Hence, fees are allowed be in the order of 0.0001 sat.

N.B :- possible change here; LN-Daemon uses millisatoshi as base unit.
"""

"""
/////////////
CONFIGURATION
/////////////
"""

NUM_NODES = 100
MAX_HOPS = 20
AMP_TIMEOUT = 0.5  # 60 seconds in c-lightning, 0.5 for now


"""
//////////////////
CORE FUNCTIONALITY
//////////////////
"""

class Node:
    """ Singular lightning node. """

    def __init__(self, id):
        self._id = id

    def update_chan(self, G, dest, amnt, htlc=False):
        """Update a channel's balance by making a payment from src to dest.

        We assume both parties sign the update automatically for simulation purposes.
        Return True if successful, False otherwise.
        """
        if G.has_edge(self, dest):
            # Assume: amnt > 0, check for available funds only
            if G[self][dest]["equity"] >= amnt:
                G[self][dest]["equity"] -= amnt
                if not htlc: G[dest][self]["equity"] += amnt
                return True
            else:
                print("Error: equity between %s and %s not available for transfer." % (self, dest))
        else:
            print("Error: no direct payment channel between %s and %s." % (self, dest))
        return False

    def _make_payment(self, G, dest, amnt, subpayment=False):
        """Make a regular single payment from this node to destination node of amnt.

        subpayment: True if part of AMP.

        Return True if successful, False otherwise.
        """
        # Reduce graph to edges with enough equity - this is very costly - fix.
        searchable = nx.DiGraph(((src, tar, attr) for src, tar, attr in G.edges(data=True) \
                                if G[src][tar]["equity"] + G[tar][src]["equity"] > amnt))

        # Finds shortest path based on lowest fees, for now.
        if self in searchable and dest in searchable and nx.has_path(searchable, self, dest):
            path = dijkstra_path(searchable, self, dest, \
                    weight=lambda u, v, d: d["fees"][0] + d["fees"][1] * amnt)
            send_amnts = calc_path_fees(G, path, amnt)

            if len(path) - 1 > MAX_HOPS:  # LN standard
                print("Error: path exceeds max-hop distance.")
                return False

            for i in range(len(path)-1):
                hop = path[i].update_chan(G, path[i+1], send_amnts[i], True)
                if hop:
                    print("Sent %f from %s to %s." % (send_amnts[i], path[i], path[i+1]))
                else:
                    print("Payment failed.")

                    # Need to reverse the HTLCs
                    for j in range(i):
                        # We know path exists from above - need to recheck if implementing closure of channels
                        G[path[i-j-1]][path[i-j]]["equity"] += send_amnts[i-j-1]
                        print("%s claimed back %f from payment to %s." % (path[i-j-1], send_amnts[i-j-1], path[i-j]))

                    return False
        else:
            print("No route available.")
            return False

        # Successful so need to release all HTLCs, so run through path again
        # But only right now if not an AMP subpayment
        if not subpayment:
            path = path[::-1]  # Reversed as secret revealed from receiver side
            for i in range(len(path)-1):
                # We know path exists from above - need to recheck if implementing closure of channels
                G[path[i]][path[i+1]]["equity"] += send_amnts[i]
                print("Released %f for %s." % (send_amnts[i], path[i+1]))

            return True
        else:
            return path, send_amnts

    def make_payment(self, G, dest, amnt, k=1):
        """Make a payment from this node to destination node of amnt.

        May be split by into k different packets [ AMP ].

        Returns True if successful, False otherwise.
        """
        if k == 1:
            return self._make_payment(G, dest, amnt)
        else:  # AMP
            amnts = [floor(amnt / k) for _ in range(k-1)]
            last = amnt - sum(amnts)
            amnts.append(last)

            subp_statuses = [False] * len(amnts)

            # @TODO: subpayments aren't actually successful here until all make it
            # So, need to have a flag at receiver that says don't release preimage.

            ttl = time.time() + AMP_TIMEOUT  # After timeout, attempt at AMP cancelled

            # Keep trying to send
            while False in subp_statuses:
                # Taking too long - n.b. all payments need to be returned!!
                # In Rusty Russell podcast - c-lightning is 60 seconds.
                if time.time() > ttl:
                    print("AMP taking too long... releasing back funds.")
                    for i, (has_paid, amnt) in enumerate(zip(subp_statuses, amnts)):
                        if has_paid:
                            path = has_paid[0][::-1]
                            send_amnts = has_paid[1][::-1]

                            # Need to reverse the HTLCs
                            for j in range(0, len(path)-1):
                                G[path[j+1]][path[j]]["equity"] += send_amnts[j]
                                print("%s claimed back %f from payment to %s." % (path[j+1], send_amnts[j], path[j]))
                    return False

                for i, (has_paid, amnt) in enumerate(zip(subp_statuses, amnts)):
                    if not has_paid:
                        res = self._make_payment(G, dest, amnt, True)
                        subp_statuses[i] = res

            # All subpayments successful, so receiver knows base preimage to all subpayments
            # Hence, need to release preimage back towards receiver on all subpayments
            for subpayment in subp_statuses:
                path = subpayment[0]
                send_amnts = subpayment[1]

                path = path[::-1]  # Reversed as secret revealed from receiver side
                for i in range(len(path)-1):
                    # We know path exists from above - need to recheck if implementing closure of channels
                    G[path[i]][path[i+1]]["equity"] += send_amnts[i]
                    print("Released %f for %s." % (send_amnts[i], path[i+1]))
            return True

    def __str__(self):
        return "Node %d" % self._id


def calc_path_fees(G, path, amnt):
    """Calculate the compound path fees required for a given path.

    Note: compounding as amnt of equity moving per node is different!
    """
    hop_amnts = [amnt]
    path = path[1:][::-1]  # No fees on first hop, reversed
    for i in range(len(path)-1):
        fees = G[path[i]][path[i+1]]["fees"]
        fee_this_hop = fees[0] + fees[1] * hop_amnts[-1]
        hop_amnts.append(fee_this_hop + hop_amnts[-1])
    return hop_amnts[::-1]


def calc_g_unbalance(G, pairs):
    """Calculate the unbalance of equity between payment channels.

    Higher equity channels take proportionally higher effect on the resultant ratio.

    Returns a float in range from 0 (completely balanced) to 1 (completely unbalanced).
    """
    total_diff = 0
    total_equity = 0
    for pair in pairs:
        u, v = pair[0], pair[1]
        total_diff += abs(G[u][v]["equity"] - G[v][u]["equity"])
        total_equity += G[u][v]["equity"] + G[v][u]["equity"]
    return total_diff / total_equity


"""
//////////
SIMULATION
//////////
"""

@nx.utils.decorators.py_random_state(3)
def generate_wattz_strogatz(n, k, p, seed=None):
    """As taken from NetworkX random_graph src code and modified.

    > SMALL WORLD graph.
    """
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.DiGraph()
    nodes = [Node(i) for i in range(n)]

    edge_pairs = set()

    # connect each node to k/2 neighbours
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        pairs = list(zip(nodes, targets))
        for pair in pairs:
            G.add_edge(pair[0], pair[1], equity=20, fees=[0.1, 0.005])
            G.add_edge(pair[1], pair[0], equity=10, fees=[0.1, 0.005])
            edge_pairs.add((pair[0], pair[1]))

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
                    G.remove_edge(v, u)
                    edge_pairs.remove((u, v))

                    G.add_edge(u, w, equity=20, fees=[0.1, 0.005])
                    G.add_edge(w, u, equity=10, fees=[0.1, 0.005])
                    edge_pairs.add((u, w))
    return G, edge_pairs

@nx.utils.decorators.py_random_state(2)
def generate_barabasi_albert(n, m, seed=None):
    """As taken from NetworkX random_graph src code and modified.

    > SCALE FREE graph.
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Target nodes for new edges
    targets = [Node(i) for i in range(m)]

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.DiGraph()
    G.add_nodes_from(targets)

    edge_pairs = set()

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        src_node = Node(source)
        # Add edges to m nodes from the source.
        pairs = list(zip([src_node] * m, targets))
        for pair in pairs:
            G.add_edge(pair[0], pair[1], equity=20, fees=[0.1, 0.005])
            G.add_edge(pair[1], pair[0], equity=10, fees=[0.1, 0.005])
            edge_pairs.add((pair[0], pair[1]))

        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([src_node] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1
    return G, edge_pairs

def test_simulator():
    # G, pairs = generate_wattz_strogatz(NUM_NODES, 6, 0.3)
    G, pairs = generate_barabasi_albert(NUM_NODES, floor(NUM_NODES / 4))
    nodes = list(G)
    a, b = nodes[0], nodes[10]
    # first = next(iter(G[a]))
    print(a.make_payment(G, b, 5, 3))
    print(calc_g_unbalance(G, pairs))
    # nx.draw(G)
    # plt.show()

if __name__ == "__main__":
    test_simulator()
