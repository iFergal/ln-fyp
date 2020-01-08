import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.generic import all_shortest_paths
from networkx.generators.random_graphs import _random_subset
import matplotlib.pyplot as plt
from math import inf, floor
import time

"""
Units of transfer are presumed to be satoshi (0.00000001 BTC) - this is the smallest unit
available on BTC - in reality, the LN supports millisatoshi for fee rounding purposes.
-- Hence, fees are allowed be in the order of 0.0001 sat.
   @TODO: need to enforce ^ and how rounding works.
"""

"""
/////////////
CONFIGURATION
/////////////
"""

NUM_NODES = 100  # Spirtes paper uses 2k
NUM_TEST_NODES = 20
MAX_HOPS = 20  # LN standard
AMP_TIMEOUT = 60  # 60 seconds in c-lightning
DEBUG = True
MERCHANT_PROB = 0.67
LATENCY_DISTRIBUTION = [0.925, 0.049, 0.026]  # For [100ms, 1s, 10s] - Sprites paper

# Consumers pay merchants - this is the chance for different role,
# i.e. merchant to pay or consumer to receive, [0, 1] - 1 means both as likely as each other
ROLE_BIAS = 0.05

"""
//////////////////
CORE FUNCTIONALITY
//////////////////
"""

class Node:
    """ Singular lightning node.

    Attributes:
        id: (int) ID of node.
        latency: (float) seconds of latency to wait before passing on payment.
        merchant: (boolean) True if a merchant, False if consumer. (consumer more likely to spend)
        spend_freq: (float) likelyhood of selection to make next payment.
        receive_freq: (float) likelyhood of selection to receive next payment.
    """

    def __init__(self, id, latency, merchant, spend_freq, receive_freq):
        self._id = id
        self._latency = latency
        self._merchant = merchant
        self._spend_freq = spend_freq
        self._receive_freq = receive_freq

    def get_id(self):
        return self._id

    def get_latency(self):
        return self._latency

    def is_merchant(self):
        return self._merchant

    def get_spend_freq(self):
        return self._spend_freq

    def get_receive_freq(self):
        return self._receive_freq

    def update_chan(self, G, dest, amnt, htlc=False, test_payment=False, debug=DEBUG):
        """Update a channel's balance by making a payment from src (self) to dest.

        We assume both parties sign the update automatically for simulation purposes.

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination (neighbour) node to attempt to send money to.
            amnt: (float) number of satoshi to send.
            htlc: (boolean) True if funds are to be held up in hash-time locked contact. (False by default)
            test_payment: (boolean) True to test if route will support amnt + fees. (False by default)
            debug: (boolean) True to display debug print statements.

        Returns:
            True if successful,
            False otherwise.
        """
        if G.has_edge(self, dest):
            # Assume: amnt > 0, check for available funds only
            if G[self][dest]["equity"] >= amnt:
                if test_payment: return True

                G[self][dest]["equity"] -= amnt
                if not htlc: G[dest][self]["equity"] += amnt

                return True
            else:
                if debug: print("Error: equity between %s and %s not available for transfer." % (self, dest))
        else:
            if debug: print("Error: no direct payment channel between %s and %s." % (self, dest))
        return False

    def _make_payment(self, G, dest, amnt, subpayment=False, test_payment=False, failed_routes=[], debug=DEBUG):
        """Make a regular single payment from this node to destination node of amnt.

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to send money to.
            amnt: (float) number of satoshi to send.
            subpayment: (boolean) True if part of an atomic multipath payment.
            test_payment: (boolean) True to test if route will support amnt + fees. (False by default)
            failed_routes: (list<list<Node>>) previously failed routes to avoid.
            debug: (boolean) True to display debug print statements.

        Returns:
            - if regular:
                True if successful,
                False otherwise.
            - if subpayment of AMP:
                the path and amounts sent if successful (for future reference) - or just True if test_payment,
                False, failed_routes otherwise (False, None if all routes exhausted or none exist).
        """
        # Reduce graph to edges with enough equity - this is very costly - fix.
        searchable = nx.DiGraph(((src, tar, attr) for src, tar, attr in G.edges(data=True) \
                                if G[src][tar]["equity"] + G[tar][src]["equity"] >= amnt))

        # Finds shortest path based on lowest fees, for now.
        if self in searchable and dest in searchable:
            paths = [p for p in all_shortest_paths(searchable, self, dest, \
                        weight=lambda u, v, d: d["fees"][0] + d["fees"][1] * amnt) \
                        if p not in failed_routes]

            if len(paths) == 0:
                if debug: print("Error: no remaining possible routes [%d]." % subpayment)
                if subpayment: return False, None  # None as there are no more routes so stop trying
                return False

            path = paths[0]

            if len(path) - 1 > MAX_HOPS:
                if debug: print("Error: path exceeds max-hop distance [%d]." % subpayment)
                failed_routes.append(path)

                if subpayment: return False, failed_routes
                return False

            send_amnts = calc_path_fees(G, path, amnt)

            # Send out payment - create HTLCs
            for i in range(len(path)-1):
                time.sleep(path[i].get_latency())  # Need to find out exactly where latency should be applied
                hop = path[i].update_chan(G, path[i+1], send_amnts[i], True, test_payment, debug=debug)
                if hop:
                    if not test_payment and debug: print("Sent %f from %s to %s. [%d]" % (send_amnts[i], path[i], path[i+1], subpayment))
                else:
                    failed_routes.append(path)

                    if not test_payment:
                        if debug: print("Error: Payment failed [%d]." % subpayment)

                        # Need to reverse the HTLCs
                        for j in range(i):
                            time.sleep(path[i-j-1].get_latency())  # Again, need to check
                            # We know path exists from above - need to recheck if implementing closure of channels
                            G[path[i-j-1]][path[i-j]]["equity"] += send_amnts[i-j-1]
                            if debug: print("%s claimed back %f from payment to %s. [%d]" % (path[i-j-1], send_amnts[i-j-1], path[i-j], subpayment))

                    if subpayment: return False, failed_routes
                    return False
        else:
            if debug: print("No route available.")
            if subpayment: return False, None  # None as there are no routes so stop trying
            return False

        if test_payment: return True

        # Successful so need to release all HTLCs, so run through path again
        # But only right now if not an AMP subpayment (and not a test payment)
        if not subpayment:
            path = path[::-1]  # Reversed as secret revealed from receiver side
            send_amnts = send_amnts[::-1]
            for i in range(len(path)-1):
                time.sleep(path[i].get_latency())  # Again, need to check - revealing R takes latency too
                # We know path exists from above - need to recheck if implementing closure of channels
                G[path[i]][path[i+1]]["equity"] += send_amnts[i]
                if debug: print("Released %f to %s." % (send_amnts[i], path[i]))

            return True
        else:
            return path, send_amnts

    def make_payment(self, G, dest, amnt, k=1, test_payment=False, debug=DEBUG):
        """Make a payment from this node to destination node of amnt.

        May be split by into k different packets [ AMP ].

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to send money to.
            amnt: (float) number of satoshi to send.
            k: (int) AMP if >1 - number of ways to split payment. (1 by default)
            test_payment: (boolean) True to test if route will support amnt + fees. (False by default)
            debug: (boolean) True to display debug print statements.

        Returns:
            True if successful,
            False otherwise.
        """
        if k == 1:
            return self._make_payment(G, dest, amnt, False, test_payment, debug=debug)
        else:  # AMP
            amnts = [floor(amnt / k) for _ in range(k-1)]
            last = amnt - sum(amnts)
            amnts.append(last)

            subp_statuses = [(False, []) for _ in range(len(amnts))]

            ttl = time.time() + AMP_TIMEOUT  # After timeout, attempt at AMP cancelled

            # Keep trying to send
            while any(False in subp for subp in subp_statuses):
                # Taking too long - n.b. all subpayments need to be returned!
                # In Rusty Russell podcast - c-lightning is 60 seconds.
                if time.time() > ttl or (False, None) in subp_statuses:
                    if debug: print("AMP taking too long... releasing back funds.")
                    for i, subp in enumerate(subp_statuses):
                        path = subp[0]
                        if path:
                            path = path[::-1]
                            send_amnts = subp[1][::-1]

                            # Need to reverse the HTLCs
                            for j in range(0, len(path)-1):
                                time.sleep(path[j+1].get_latency())  # Again latency to reveal R
                                G[path[j+1]][path[j]]["equity"] += send_amnts[j]
                                if debug: print("%s claimed back %f from payment to %s." % (path[j+1], send_amnts[j], path[j]))
                    return False

                for i, (subp, amnt) in enumerate(zip(subp_statuses, amnts)):
                    if not subp[0]:
                        res = self._make_payment(G, dest, amnt, i+1, False, subp[1], debug=debug)
                        subp_statuses[i] = res

            # All subpayments successful, so receiver knows base preimage to all subpayments
            # Hence, need to release preimage back towards receiver on all subpayments
            for subpayment in subp_statuses:
                path = subpayment[0]
                send_amnts = subpayment[1]

                path = path[::-1]  # Reversed as secret revealed from receiver side
                send_amnts = send_amnts[::-1]
                for i in range(len(path)-1):
                    time.sleep(path[i].get_latency())  # Reveal R latency
                    # We know path exists from above - need to recheck if implementing closure of channels
                    G[path[i]][path[i+1]]["equity"] += send_amnts[i]
                    if debug: print("Released %f from HTLC to %s." % (send_amnts[i], path[i]))
            return True

    def get_total_equity(self, G):
        """Returns the total equity held by a node (not locked up in HTLCs)."""
        out_edges = G.out_edges(self)
        total = 0

        for out_edge in out_edges:
            total += G[self][out_edge[1]]["equity"]

        return total

    def __str__(self):
        return "Node %d" % self._id


def calc_path_fees(G, path, amnt):
    """Calculate the compound path fees required for a given path.

    Note: compounding as amnt of equity moving per node is different!

    Args:
        path: (list<Node>) path for which fees are to be calculated.
        amnt: (float) number of satoshi to send to final Node in path.

    Returns:
        a list of satoshi to send at each hop.
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

    Args:
        pairs: represents payment channels - pairs of nodes (but only one direction so no duplicates)

    Returns:
        a float in range from 0 (completely balanced) to 1 (completely unbalanced).
    """
    total_diff = 0
    total_equity = 0
    for pair in pairs:
        u, v = pair[0], pair[1]
        total_diff += abs(G[u][v]["equity"] - G[v][u]["equity"])
        total_equity += G[u][v]["equity"] + G[v][u]["equity"]
    return total_diff / total_equity

def graph_str(G):
    """Return a string representation of graph. """
    str = "<-----START GRAPH----->\n"
    nodes = list(G)
    for node in nodes:
        str += "[%d] => [" % node.get_id()
        out_edges = G.out_edges(node)
        for out_edge in out_edges:
            str += " {%d/%.2f}" % (out_edge[1].get_id(), G[out_edge[0]][out_edge[1]]["equity"])
        str += " ]\n"
    str += "<-----END GRAPH----->"
    return(str)


"""
//////////
SIMULATION
//////////
"""

def init_random_node(i):
    """Initialise and return a new node based on given simulation parameters, with given ID i. """
    merchant = np.random.choice([True, False], 1, p=[MERCHANT_PROB, 1 - MERCHANT_PROB])
    latency = np.random.choice([0.1, 1, 10], 1, p=LATENCY_DISTRIBUTION)
    spend_freq = 1  # unsure how to handle freqs yet
    receive_freq = 1

    return Node(i, latency, merchant, spend_freq, receive_freq)

def generate_edge_args():
    """Generate random equity and fee distribution arguments for a new pair of edges. """
    dir_a = [20, [0.1, 0.005]]
    dir_b = [10, [0.1, 0.005]]

    return [dir_a, dir_b]

@nx.utils.decorators.py_random_state(3)
def generate_wattz_strogatz(n, k, p, seed=None, test=False):
    """As taken from NetworkX random_graph src code and modified.

    > SMALL WORLD graph.

    Args:
        n: (int) the number of nodes in the generated graph.
        k: (int) each node is connected to k nearest neighbours in ring topology.
        p: (float) probability of rewiring each edge.
        seed: (int) seed for RNG. (None by default)
        test: (boolean) True if graph to be generated is for the test function. (False by default)

    Returns:
        the generated graph G.
    """
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.DiGraph()
    if test:
        nodes = [Node(i, 0, False, 0, 0) for i in range(n)]
    else:
        nodes = [init_random_node(i) for i in range(n)]

    edge_pairs = set()

    # Connect each node to k/2 neighbours
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # First j nodes are now last in list
        pairs = list(zip(nodes, targets))
        for pair in pairs:
            if test:
                G.add_edge(pair[0], pair[1], equity=20, fees=[0.1, 0.005])
                G.add_edge(pair[1], pair[0], equity=10, fees=[0.1, 0.005])
            else:
                args = generate_edge_args()
                G.add_edge(pair[0], pair[1], equity=args[0][0], fees=args[0][1])
                G.add_edge(pair[1], pair[0], equity=args[1][0], fees=args[1][1])
            edge_pairs.add((pair[0], pair[1]))

    # Rewire edges from each node
    # Loop over all nodes in order (label) and neighbours in order (distance)
    # No self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # Outer loop is neighbours
        targets = nodes[j:] + nodes[0:j]  # First j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # Skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.remove_edge(v, u)
                    edge_pairs.remove((u, v))

                    if test:
                        G.add_edge(u, w, equity=20, fees=[0.1, 0.005])
                        G.add_edge(w, u, equity=10, fees=[0.1, 0.005])
                    else:
                        args = generate_edge_args()
                        G.add_edge(u, w, equity=args[0][0], fees=args[0][1])
                        G.add_edge(w, u, equity=args[1][0], fees=args[1][1])
                    edge_pairs.add((u, w))
    return G, edge_pairs

@nx.utils.decorators.py_random_state(2)
def generate_barabasi_albert(n, m, seed=None):
    """As taken from NetworkX random_graph src code and modified.

    > SCALE FREE graph.

    Args:
        n: (int) the number of nodes in the generated graph.
        m: (int) the number of edges to attach from a new node to existing nodes.
        seed: (int) seed for RNG. (None by default)

    Returns:
        the generated graph G.
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Target nodes for new edges
    targets = []
    for i in range(m):
        targets.append(init_random_node(i))

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.DiGraph()
    G.add_nodes_from(targets)

    edge_pairs = set()

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        src_node = init_random_node(source)
        # Add edges to m nodes from the source.
        pairs = list(zip([src_node] * m, targets))
        for pair in pairs:
            args = generate_edge_args()
            G.add_edge(pair[0], pair[1], equity=args[0][0], fees=args[0][1])
            G.add_edge(pair[1], pair[0], equity=args[1][0], fees=args[1][1])

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

def simulator():
    """Main simulator calling function. """
    G, pairs = generate_wattz_strogatz(NUM_NODES, 6, 0.3)
    # G, pairs = generate_barabasi_albert(NUM_NODES, floor(NUM_NODES / 4))
    nodes = list(G)
    a, b = nodes[0], nodes[10]

    # nx.draw(G)
    # plt.show()


"""
//////////////////////////
CORE FUNCTIONALITY TESTING
//////////////////////////
"""

def test_func():
    TEST_DEBUG = False

    ################## func: update_chan #######################
    # Always produces same graph due to seed
    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    # Path exists from 0 -> 2
    assert nodes[2].get_total_equity(G) == 80

    # Straight transfer
    nodes[0].update_chan(G, nodes[2], 5, debug=TEST_DEBUG)
    assert nodes[0].get_total_equity(G) == 65
    assert nodes[2].get_total_equity(G) == 85

    # HTLC
    nodes[0].update_chan(G, nodes[2], 5, htlc=True, debug=TEST_DEBUG)
    assert nodes[0].get_total_equity(G) == 60
    assert nodes[2].get_total_equity(G) == 85

    # Test payment
    nodes[2].update_chan(G, nodes[0], 5, test_payment=True, debug=TEST_DEBUG)
    assert nodes[0].get_total_equity(G) == 60
    assert nodes[2].get_total_equity(G) == 85

    # Not enough equity
    assert nodes[0].update_chan(G, nodes[2], 20, debug=TEST_DEBUG) == False

    ################## func: _make_payment [NON-AMP] ###########
    # Re-init just in case
    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    # Payment from 0 to 14 will find path 0-1-14
    path = [nodes[0], nodes[1], nodes[14]]
    amnts = calc_path_fees(G, path, 10)

    # Test payment
    e_0_1 = G[nodes[0]][nodes[1]]["equity"]
    e_1_0 = G[nodes[1]][nodes[0]]["equity"]
    e_1_14 = G[nodes[1]][nodes[14]]["equity"]
    e_14_1 = G[nodes[14]][nodes[1]]["equity"]

    # When k=1, make_payment calls _make_payment directly
    nodes[0]._make_payment(G, nodes[14], 10, test_payment=True, debug=TEST_DEBUG)

    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1

    # Real payment
    nodes[0]._make_payment(G, nodes[14], 10, debug=TEST_DEBUG)

    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts[0]
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + amnts[0]
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts[1]
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1 + amnts[1]

    # There's no more equity from 1 to 14, so try this route again to test
    # for funds being released back correctly when a route fails mid-way forward.
    nodes[0]._make_payment(G, nodes[14], 5, debug=TEST_DEBUG)

    # Should be the same as payment only made it one hop.
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts[0]
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + amnts[0]
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts[1]
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1 + amnts[1]

    # No route available
    assert nodes[0]._make_payment(G, nodes[14], 100, debug=TEST_DEBUG) == False

    # One-hop payments - make sure finds direct path.
    path = [nodes[11], nodes[9]]
    amnts = calc_path_fees(G, path, 5)

    e_11_9 = G[nodes[11]][nodes[9]]["equity"]
    e_9_11 = G[nodes[9]][nodes[11]]["equity"]

    nodes[11]._make_payment(G, nodes[9], 5, debug=TEST_DEBUG)

    assert G[nodes[11]][nodes[9]]["equity"] == e_11_9 - amnts[0]
    assert G[nodes[9]][nodes[11]]["equity"] == e_9_11 + amnts[0]


    ################## func: make_payment [AMP] ################
    # Re-init just in case
    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    # From above, we've already tested that single payments work and if they fail revert correctly
    # So, for AMP - need to check:
    #   1) split payment that works straight off
    #   2) fails, retries routes and works
    #   3) fails overall with subpayments that didn't fail reverting

    # Works with no fails - right now, this means it tries same route and keeps working.
    # Again 0-1-14 sent as packets of 3/3/4
    path = [nodes[0], nodes[1], nodes[14]]
    amnts = calc_path_fees(G, path, 3)
    amnts_2 = calc_path_fees(G, path, 4)

    e_0_1 = G[nodes[0]][nodes[1]]["equity"]
    e_1_0 = G[nodes[1]][nodes[0]]["equity"]
    e_1_14 = G[nodes[1]][nodes[14]]["equity"]
    e_14_1 = G[nodes[14]][nodes[1]]["equity"]

    nodes[0].make_payment(G, nodes[14], 10, k=3, test_payment=True, debug=TEST_DEBUG)

    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts[0] * 2 - amnts_2[0]
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + amnts[0] * 2 + amnts_2[0]
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts[1] * 2 - amnts_2[1]
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1 + amnts[1] * 2 + amnts_2[1]

    # Try same payment again - will work for subpayments but last payment will fail
    # First 2 subpayments reverted - so should be same as above.
    # Stops when all routes exhausted - will also stop if taking too long - no need to test this.
    # @TODO: If better pathfinding implemented this will probably break.
    nodes[0].make_payment(G, nodes[14], 10, k=3, test_payment=True, debug=TEST_DEBUG)

    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts[0] * 2 - amnts_2[0]
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + amnts[0] * 2 + amnts_2[0]
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts[1] * 2 - amnts_2[1]
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1 + amnts[1] * 2 + amnts_2[1]

    # Try a payment that has failed routes, but finds enough other routes to
    # successfully route the payment - reset so we can use same route.
    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    # Sent as 5/5/5 - alternative route 0-19-14 for last payment.
    path = [nodes[0], nodes[1], nodes[14]]
    amnts = calc_path_fees(G, path, 5)

    path = [nodes[0], nodes[19], nodes[14]]
    amnts_2 = calc_path_fees(G, path, 5)

    e_0_1 = G[nodes[0]][nodes[1]]["equity"]
    e_1_0 = G[nodes[1]][nodes[0]]["equity"]
    e_1_14 = G[nodes[1]][nodes[14]]["equity"]
    e_14_1 = G[nodes[14]][nodes[1]]["equity"]
    e_0_19 = G[nodes[0]][nodes[19]]["equity"]
    e_19_0 = G[nodes[19]][nodes[0]]["equity"]
    e_19_14 = G[nodes[19]][nodes[14]]["equity"]
    e_14_19 = G[nodes[14]][nodes[19]]["equity"]

    nodes[0].make_payment(G, nodes[14], 15, k=3, test_payment=True, debug=TEST_DEBUG)

    # 2 on 0-1-14, 1 on 0-19-14
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts[0] * 2
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + amnts[0] * 2
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts[1] * 2
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1 + amnts[1] * 2
    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - amnts_2[0]
    assert G[nodes[19]][nodes[0]]["equity"] == e_19_0 + amnts_2[0]
    assert G[nodes[19]][nodes[14]]["equity"] == e_19_14 - amnts_2[1]
    assert G[nodes[14]][nodes[19]]["equity"] == e_14_19 + amnts_2[1]

if __name__ == "__main__":
    test_func()

simulator()
