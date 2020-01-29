import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.generic import all_shortest_paths
from networkx.algorithms.simple_paths import PathBuffer
from networkx.generators.random_graphs import _random_subset
import matplotlib.pyplot as plt
from math import inf, floor
from queue import PriorityQueue, Empty
from uuid import uuid4
from heapq import heappush, heappop
from itertools import count
import time

"""
Units of transfer are presumed to be satoshi (0.00000001 BTC) - this is the smallest unit
available on BTC - in reality, the LN supports millisatoshi for fee rounding purposes.
-- Hence, fees are allowed be in the order of 0.0001 sat.
"""

"""
/////////////
CONFIGURATION
/////////////
"""

NUM_NODES = 200  # Spirtes paper uses 2k
NUM_TEST_NODES = 20
MAX_HOPS = 20  # LN standard
AMP_TIMEOUT = 60  # 60 seconds in c-lightning
DEBUG = False
NUM_STARTING_PAYMENTS = 50
MERCHANT_PROB = 0.67
LATENCY_OPTIONS = [0.1, 1, 10]  # Sprites paper, in seconds
LATENCY_DISTRIBUTION = [1, 0, 0] #0.925, 0.049, 0.026]
CTLV_OPTIONS = [9]  # Default on LN for now
CTLV_DISTRIBUTION = [1]
CLTV_MULTIPLIER = 600  # BTC block time is ~10 minutes, so 600 seconds
DELAY_RESP_VAL = [0, 0.2, 0.4, 0.6, 0.8, 1]  # Amount of timelock time node will wait before cancelling/claiming
DELAY_RESP_DISTRIBUTION = [1, 0, 0, 0, 0, 0]#0.9, 0.05, 0.01, 0.005, 0.005, 0.03]
MAX_PATH_ATTEMPTS = 30  # For AMP, if we try a subpayment on this many paths and fail, stop

# Consumers pay merchants - this is the chance for different role,
# i.e. merchant to pay or consumer to receive, [0, 0.5] - 0.5 means both as likely as each other
ROLE_BIAS = 0.05

# Terminal output colours
class bcolours:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# Analytics
success_count = 0
fail_count = 0
amp_retry_count = 0

"""
//////////////////
CORE FUNCTIONALITY
//////////////////
"""

class Packet:
    """Passes between nodes to send equity down a path, and release HTLCs back.

    Index mapping is used to simulate onion routing.

    Attributes:
        path: (list<Node>) list of nodes on the path.
        index_mapping: (dict) dictionary mapping Node ID's to indexes.
        send_amnts: (list<float>) list of amnts to attempt to send per hop on onward direction.
        type: (string) type of communication packet.
            - pay: directly update channel balance between neighbouring nodes.
            - pay_htlc: lock up funds to be sent in HTLC between neighbouring nodes.
            - preimage: release HTLC, revealing preimage (R) to neighbouring node.
            - cancel: cancel previously set up HTLC between neighbouring node.
            - cancel_rest: same as cancel, but no HTLC between previous node was set up yet.
        subpayment: (boolean / (UUID, int, int, float))
                - (ID of containing AMP, index of subpayment, total num subpayments, timeout time),
                - or False if not part of AMP.
    """

    def __init__(self, path, index_mapping, send_amnts, type="pay_htlc", subpayment=False):
        self._path = path
        self._index_mapping = index_mapping
        self._type = type
        self._send_amnts = send_amnts
        self._subpayment = subpayment
        self._htlc_ids = []
        self._timestamp = None
        self._final_index = len(path) - 1
        self._delayed = False

    def get_path(self):
        return self._path

    def get_node(self, index):
        return self._path[index]

    def get_index(self, ID):
        return self._index_mapping[ID]

    def get_type(self):
        return self._type

    def set_type(self, type):
        self._type = type

    def get_total_fees(self):
        return sum([(self._send_amnts[i] - self._send_amnts[-1]) for i in range(len(self._send_amnts))])

    def get_amnt(self, index):
        return self._send_amnts[index]

    def get_subpayment(self):
        return self._subpayment

    def get_timestamp(self):
        return self._timestamp

    def set_timestamp(self, timestamp):
        self._timestamp = timestamp

    def get_final_index(self):
        return self._final_index

    def add_htlc_id(self, id):
        self._htlc_ids.append(id)

    def get_htlc_id(self, index):
        return self._htlc_ids[index]

    def is_delayed(self):
        return self._delayed

    def set_delayed(self):
        self._delayed = True

    def __lt__(self, other):
        return self.get_timestamp() < other.get_timestamp()


class Node:
    """Singular lightning node.

    Attributes:
        id: (int) ID of node.
        merchant: (boolean) True if a merchant, False if consumer. (consumer more likely to spend)
        spend_freq: (float) likelihood of selection to make next payment.
        receive_freq: (float) likelihood of selection to receive next payment.
    """

    def __init__(self, id, merchant, spend_freq, receive_freq):
        self._id = id
        self._merchant = merchant
        self._spend_freq = spend_freq
        self._receive_freq = receive_freq
        self._amp_out = {}  # Outgoing AMPs still active
        self._amp_in = {}  # Awaiting incoming AMPs
        self._htlc_out = {}  # Maps other nodes to HTLCs
        self._queue = PriorityQueue()  # Packet queue

    def get_id(self):
        return self._id

    def is_merchant(self):
        return self._merchant

    def get_spend_freq(self):
        return self._spend_freq

    def get_receive_freq(self):
        return self._receive_freq

    def update_htlc_out(self, node, id, amnt, ttl):
        """Neighbouring nodes can add new HTLC info when established on their channel.

        Args:
            node: (Node) the node at the other end of the channel.
            id: (UUID) ID of HTLC between nodes.
            amnt: (float) number of satoshi contained in the HTLC.
            ttl: (float) real UTC time when the HTLC will be available to claim by self.
        """
        if not node in self._htlc_out:
            self._htlc_out[node] = {}

        self._htlc_out[node][id] = (amnt, ttl)

    def get_queue(self):
        return self._queue

    def receive_packet(self, packet, debug=DEBUG):
        """Receive a packet from the main node communicator.

        Args:
            packet: (Packet) the packet to add to the Node's priority queue.
            debug: (boolean) True to display debug print statements.
        """
        if debug: print(bcolours.OKGREEN + "%s is receiving packet from node communicator." % self + bcolours.ENDC)
        self._queue.put(packet)

    def process_next_packet(self, G, node_comm, test_mode=False, debug=DEBUG):
        """Process next packet if any.

        Args:
            G: NetworkX graph in use.
            debug: (boolean) True to display debug print statements.
            node_comm: (NodeCommunicator) main communicator to send packets to.
            test_mode: (boolean) True if running from test function.
            debug: (boolean) True to display debug print statements.
        """
        # Analytics
        global success_count
        global fail_count
        global amp_retry_count

        # Need to only remove from the PriorityQueue if ready to be proccessed time-wise !
        try:
            p = self._queue.get(False)

            if time.time() < p.get_timestamp():  # Not ready to be removed, return to queue
                self._queue.put(p)
                # if debug: print(bcolours.WARNING + "No packets are ready to process at %s yet." % self + bcolours.ENDC)
                return

            index = p.get_index(self._id)
            type = p.get_type()
            amnt = p.get_amnt(index - 1)
            subpayment = p.get_subpayment()

            # -- ORDER --
            # Equity moves when destination node receives message from source node.
            # HTLC equity is claimed back by source node as they receive preimage, and then sent to next.
            # HTLCs are fully cancelled when cancel message is received by destination node.

            if type == "pay" or type == "pay_htlc":
                src = p.get_node(index - 1)
                # Assume: amnt > 0, check for available funds only
                if G[src][self]["equity"] >= amnt:
                    G[src][self]["equity"] -= amnt  # To HTLC
                    if type == "pay":
                        G[self][src]["equity"] += amnt
                        if debug: print("Sent %.4f from %s to %s." % (amnt, src, self) + (" [%d]" % subpayment[1] if subpayment else ""))
                    else:
                        id = uuid4()
                        ttl = time.time() + node_comm.get_ctlv_delta(src, self) * CLTV_MULTIPLIER
                        src.update_htlc_out(self, id, amnt, ttl)

                        p.add_htlc_id(id)

                        if debug: print("Sent (HTLC) %.4f from %s to %s." % (amnt, src, self) + (" [%d]" % subpayment[1] if subpayment else ""))
                        if index == p.get_final_index():
                            # Receiver - so release preimage back and claim funds if not AMP
                            if not subpayment:
                                G[self][src]["equity"] += amnt

                                dest = p.get_node(index - 1)
                                p.set_type("preimage")

                                if debug: print("Received payment - sending preimage release message to node communicator from %s (-> %s)." % (self, dest))
                                node_comm.send_packet(self, dest, p)
                            else:  # If AMP, we need to keep and store all the partial payments
                                id, subp_index, _, ttl = p.get_subpayment()
                                if not id in self._amp_in:
                                    self._amp_in[id] = []

                                if time.time() <= ttl:  # Within time
                                    self._amp_in[id].append(p)
                                    if debug: print(bcolours.OKGREEN + "Received partial payment at %s. [%d]" % (self, subpayment[1]) + bcolours.ENDC)

                                    if len(self._amp_in[id]) == subpayment[2]:  # All collected, release HTLCs
                                        for s in self._amp_in[id]:
                                            index = s.get_index(self._id)
                                            dest = s.get_node(index - 1)
                                            amnt = s.get_amnt(index - 1)
                                            G[self][dest]["equity"] += amnt

                                            s.set_type("preimage")
                                            if debug: print("Sending preimage release message to node communicator from %s (-> %s). [%d]" % (self, dest, subpayment[1]))
                                            node_comm.send_packet(self, dest, s)
                                        del self._amp_in[id]
                                else:  # Out of time, need to release subpayments back
                                    for s in self._amp_in[id]:
                                        index = s.get_index(self._id)
                                        dest = s.get_node(index - 1)
                                        amnt = s.get_amnt(index - 1)
                                        G[self][dest]["equity"] += amnt

                                        s.set_type("cancel")
                                        if debug: print("Sending cancel message to node communicator from %s (-> %s). [%d]" % (self, dest, subpayment[1]))
                                        node_comm.send_packet(self, dest, s)
                        else:
                            # Need to keep sending it on, but only if funds are available
                            dest = p.get_node(index + 1)
                            amnt = p.get_amnt(index)
                            if G[self][dest]["equity"] >= amnt:
                                if debug: print("Sending pay_htlc message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                                node_comm.send_packet(self, dest, p)
                            else:
                                if debug: print(bcolours.FAIL + "Error: equity between %s and %s not available for transfer - reversing." % (src, self) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
                                p.set_type("cancel")
                                node_comm.send_packet(self, p.get_node(index - 1), p)
                else:
                    if debug: print(bcolours.FAIL + "Error: equity between %s and %s not available for transfer." % (src, self) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)

                    dest = p.get_node(index - 1)  # Propagate back CANCEL
                    p.set_type("cancel_rest")  # Not regular cancel as no HTLC with prev. node.
                    if debug: print("Sending cancellation message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                    node_comm.send_packet(self, dest, p)
            elif type == "preimage":
                # Release HTLC entry
                del self._htlc_out[p.get_node(index + 1)][p.get_htlc_id(index)]

                if index != 0:  # Keep releasing
                    dest = p.get_node(index - 1)
                    G[self][dest]["equity"] += amnt

                    if debug:
                        print("%s claimed back %.4f from payment to %s." % (self, amnt, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                        print("Sending preimage release message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                    node_comm.send_packet(self, dest, p)
                else:  # Sender, receipt come back
                    if not subpayment:
                        success_count += 1

                    if debug:
                        j = p.get_final_index()
                        print(bcolours.OKGREEN + "Payment [%s -> %s // %.4f] successly completed." % (self, p.get_node(j), p.get_amnt(j-1)) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)

                    if p.get_subpayment() and p.get_subpayment()[0] in self._amp_out:
                        # Then, receiver must have got all of AMP
                        del self._amp_out[p.get_subpayment()[0]]
                        success_count += 1

                        j = p.get_final_index()
                        if debug: print(bcolours.OKGREEN + "AMP from %s to %s [%.4s] fully completed." % (self, p.get_node(j), p.get_amnt(j-1)) + bcolours.ENDC)
            else:  # Cancel message
                dest = p.get_node(index + 1)
                amnt = p.get_amnt(index)

                if not p.is_delayed():
                    cancel_chance = np.random.choice(DELAY_RESP_VAL, 1, p=DELAY_RESP_DISTRIBUTION)[0] if not test_mode else 0

                    # Some unresponsive or malicious nodes might delay the cancellation
                    if cancel_chance:
                        # Simulate this by requeuing the Packet with an updated timestamp to open
                        ttl = time.time() + node_comm.get_ctlv_delta(self, dest) * CLTV_MULTIPLIER * cancel_chance
                        p.set_timestamp(ttl)
                        p.set_delayed()
                        self._queue.put(p)

                        if debug: print("%s is waiting for %.4f of HTLC timeout before signing..." % (self, cancel_chance))
                        return

                if p.get_type() == "cancel":  # Not cancel_rest
                    del self._htlc_out[dest][p.get_htlc_id(index)]

                    G[self][dest]["equity"] += amnt  # Claim back
                    if debug: print("%s cancelled HTLC and claimed back %.4f from payment to %s." % (self, amnt, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                else:
                    p.set_type("cancel")

                if index == 0 and debug:
                    j = p.get_final_index()
                    print(bcolours.FAIL + "Payment [%s -> %s // %.4f] failed and returned." % (self, p.get_node(j), p.get_amnt(j-1)) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)

                if index != 0:  # Send cancel message on
                    dest = p.get_node(index - 1)
                    if debug: print("Sending cancellation message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                    node_comm.send_packet(self, dest, p)
                elif subpayment:  # Partial payment of AMP failed
                    # subpayment in form [ID, index, total num, ttl]
                    id, i, k, ttl = subpayment

                    if id in self._amp_out:  # Still on-going
                        if time.time() < ttl and self._amp_out[id][2]:
                            # Still within time limit - so try again!
                            # In Rusty Russell podcast - c-lightning is 60 seconds.
                            j = p.get_final_index()
                            self._amp_out[id][0][i].append((p.get_total_fees(), p.get_path()))
                            new_subp = (id, i, k)

                            if debug: print(bcolours.WARNING + "Resending... [%d]" % i + bcolours.ENDC)
                            self._init_payment(G, p.get_node(j), p.get_amnt(j-1), node_comm, new_subp, test_mode=test_mode, debug=debug)

                            amp_retry_count += 1
                        else:
                            fail_count += 1
                            del self._amp_out[id]

                            if debug:
                                j = p.get_final_index()
                                print(bcolours.FAIL + "Partial payment [%d] of failed AMP from %s to %s [%.4s] returned - not resending." % (i, self, p.get_node(j), p.get_amnt(j-1)) + bcolours.ENDC)
                    else:
                        if debug:
                            j = p.get_final_index()
                            print(bcolours.FAIL + "Partial payment [%d] of failed AMP from %s to %s [%.4s] returned - not resending." % (i, self, p.get_node(j), p.get_amnt(j-1)) + bcolours.ENDC)
                else:  # Non-subpayment failed
                    fail_count += 1
        except Empty:
            if debug and False: print(bcolours.WARNING + "No packets to process at %s." % self + bcolours.ENDC)

    def clear_old_amps(self, G, node_comm, force=False, debug=DEBUG):
        """Check for timed out AMP payments with partial subpayments that need
        to be released back to the sender.

        Args:
            G: NetworkX graph in use.
            node_comm: (NodeCommunicator) main communicator to send packets to.
            force: (boolean) clear all partially received payments regardless of timeout - for testing.
        """
        to_delete = []
        for id in self._amp_in:
            if force or p.get_subpayment()[3] - time.time() < 0:
                if debug: print(bcolours.WARNING + "Cancelling partial subpayments from %s for AMP ID %s" % (self, id) + bcolours.ENDC)
                for p in self._amp_in[id]:
                    index = p.get_index(self._id)
                    dest = p.get_node(index - 1)
                    amnt = p.get_amnt(index - 1)

                    p.set_type("cancel")
                    if debug: print("Sending cancel message to node communicator from %s (-> %s)." % (self, dest))
                    node_comm.send_packet(self, dest, p)
                to_delete.append(id)

        for id in to_delete:
            del self._amp_in[id]

    def send_direct(self, G, dest, amnt, node_comm, debug=DEBUG):
        """Send funds over a direct payment channel to neighbouring node.

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to find best path to.
            amnt: (float) number of satoshi the resultant path must support.
            node_comm: (NodeCommunicator) main communicator to send packets to.
            debug: (boolean) True to display debug print statements.

        Returns:
            True if payment packet sent out successfully,
            False otherwise.
        """
        if G.has_edge(self, dest):
            path = [self, dest]
            index_mapping = {path[i].get_id(): i for i in range(len(path))}

            p = Packet(path, index_mapping, [amnt], "pay", False)
            node_comm.send_packet(self, dest, p)

            if debug: print("Sending pay message to node communicator from %s (-> %s)." % (self, dest))
            return True
        else:
            if debug: print(bcolours.FAIL + "Error: no direct payment channel between %s and %s." % (self, dest) + bcolours.ENDC)
            return False

    def _find_path(self, G, dest, amnt, failed_paths):
        """Attempt to find the next shortest path from self to dest within graph G, that is not in failed_paths.

        Adapted from NetworkX shorest_simple_paths src code.

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to find best path to.
            amnt: (float) number of satoshi the resultant path must support.
            failed_paths: (list) previously yielded routes to skip.

        Returns:
            a generator that produces a list of possible paths, from best to worst.

        Raises:
            NetworkXError: if self or dest are not in the input graph.
            NetworkXNoPath: if no other paths exists from src to dest.
        """
        if self not in G:
            raise nx.NodeNotFound("Src [%s] not in graph" % self)

        if dest not in G:
            raise nx.NodeNotFound("Dest [%s] not in graph" % dest)

        def length_func(path):
            send_amnts = calc_path_fees(G, path, amnt)
            return send_amnts[0] - amnt

        shortest_path_func = _dijkstra_reverse

        listA = []  # Previously yielded paths
        listB = PathBuffer()

        for p in failed_paths:
            listA.append(p[1])
            listB.push(p[0], p[1])
            listB.pop()

        if len(listA) > 0:
            prev_path = listA[-1]
        else:
            prev_path = None

        # Find one new path per func call, up until a global maximum.
        while len(listA) <= len(failed_paths) and len(listA) < MAX_PATH_ATTEMPTS:
            if not prev_path:
                length, path = shortest_path_func(G, self, dest, amnt)
                listB.push(length, path)
            else:
                ignore_nodes = set()
                ignore_edges = set()
                for i in range(1, len(prev_path)):
                    root = prev_path[:i]
                    for path in listA:
                        if path[:i] == root:
                            ignore_edges.add((path[i - 1], path[i]))
                    try:
                        spur_attempt = shortest_path_func(G, root[-1], dest, amnt,
                                                          ignore_nodes=ignore_nodes,
                                                          ignore_edges=ignore_edges)
                        if spur_attempt:
                            length, spur = spur_attempt
                            path = root[:-1] + spur
                            listB.push(length_func(path), path)
                    except nx.NetworkXNoPath:
                        pass
                    ignore_nodes.add(root[-1])

            if listB:
                path = listB.pop()
                yield path
                listA.append(path)
                prev_path = path
            else:
                raise nx.NetworkXNoPath()

    def _init_payment(self, G, dest, amnt, node_comm, subpayment=False, test_mode=False, debug=DEBUG):
        """Initialise a regular single payment from this node to destination node of amnt.

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to send money to.
            amnt: (float) number of satoshi to send.
            node_comm: (NodeCommunicator) main communicator to send packets to.
            subpayment: (boolean / (UUID, int, int))
                - (ID of subpayment group, index of subpayment, total num subpayments),
                - or False if not AMP.
            test_mode: (boolean) True if running from test function.
            debug: (boolean) True to display debug print statements.

        Returns:
            True if payment packet sent out successfully,
            False otherwise.
        """
        # Analytics
        global fail_count

        if subpayment:
            id, i, n = subpayment
            failed_routes = self._amp_out[id][0][i]

            paths = [p for p in self._find_path(G, dest, amnt, failed_routes)]
        else:
            # Only try best path once for regular payments
            paths = [_dijkstra_reverse(G, self, dest, amnt)]

        if paths[0] is False or len(paths) == 0:
            if debug: print(bcolours.FAIL + "Error: no possible routes available." + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
            if subpayment:
                self._amp_out[id][2] = False
                if debug: print(bcolours.FAIL + "AMP from %s to %s [%.4s] failed." % (self, dest, amnt) + bcolours.ENDC)
            else:
                fail_count += 1
            return False

        path = paths[0] if subpayment else paths[0][1]

        if len(path) - 1 > MAX_HOPS:
            if debug: print(bcolours.FAIL + "Error: path exceeds max-hop distance" + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
            if subpayment: self._amp_out[id][0][i].append(path)
            return False

        send_amnts = calc_path_fees(G, path, amnt)
        index_mapping = {path[i].get_id(): i for i in range(len(path))}
        type = "pay_htlc" if len(path) > 2 else "pay"
        if subpayment: ttl = self._amp_out[subpayment[0]][1]

        p = Packet(path, index_mapping, send_amnts, type,
            (subpayment[0], subpayment[1], subpayment[2], ttl) if subpayment else False)

        if debug: print("Sending %s message to node communicator from %s (-> %s)." % (type, self, path[1]) + (" [%d]" % subpayment[1] if subpayment else ""))
        node_comm.send_packet(self, path[1], p)

        return True

    def init_payment(self, G, dest, amnt, node_comm, k=1, test_mode=False, debug=DEBUG):
        """Initialse a payment from this node to destination node of amnt.

        May be split by into k different packets [ AMP ].

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to send money to.
            amnt: (float) number of satoshi to send.
            node_comm: (NodeCommunicator) main communicator to send packets to.
            k: (int) AMP if >1 - number of ways to split payment. (1 by default)
            test_mode: (boolean) True if running from test function.
            debug: (boolean) True to display debug print statements.
        """
        if k == 1:
            self._init_payment(G, dest, amnt, node_comm, False, test_mode=test_mode, debug=debug)
        else:  # AMP
            amnts = [floor(amnt / k) for _ in range(k-1)]
            last = amnt - sum(amnts)
            amnts.append(last)

            # Of form, [subpayment index, failed routes]
            subp_statuses = [[] for i in range(len(amnts))]
            id = uuid4()

            # After timeout, attempt at AMP cancelled
            ttl = time.time() + AMP_TIMEOUT
            self._amp_out[id] = [subp_statuses, ttl, True]  # True here means hasn't failed yet

            # Send off each subpayment - first attempt.
            for i in range(len(amnts)):
                self._init_payment(G, dest, amnts[i], node_comm, (id, i, k), test_mode=test_mode, debug=debug)

    def get_total_equity(self, G):
        """Returns the total equity held by a node (not locked up in HTLCs). """
        out_edges = G.out_edges(self)
        total = 0

        for out_edge in out_edges:
            total += G[self][out_edge[1]]["equity"]

        return total

    def get_largest_outgoing_equity(self, G):
        """Returns the largest equity held by the node in a single payment channel. """
        out_edges = G.out_edges(self)
        largest = 0

        for out_edge in out_edges:
            if G[self][out_edge[1]]["equity"] > largest:
                largest = G[self][out_edge[1]]["equity"]

        return largest

    def __str__(self):
        return "Node %d" % self._id


class NodeCommunicator:
    """Handles communication between nodes.

    Messages are synced to UTC time and network latency is applied by using future timestamps.
    Edge latencies between nodes are initialsed here.
    CTLV expiracy deltas (BOLT 7) per directed edge initialised here. (unit: number of BTC blocks)

    Attributes:
        nodes: (list<Node>) set of nodes to handle communication between.
        edge_pairs: (set<tuple<networkX.Graph.Edge>>) payment channels: corresponding edges grouped.
        test_mode: (boolean) True if testing - remove randomised latencies.
        debug: (boolean) True to display debug print statements.
    """
    def __init__(self, nodes, edge_pairs, test_mode=False, debug=DEBUG):
        self._nodes = nodes
        self._edge_pairs = edge_pairs
        self._debug = debug
        self._latencies = {}
        self._ctlv_deltas = {}

        for edge_pair in edge_pairs:
            latency = np.random.choice(LATENCY_OPTIONS, 1, p=LATENCY_DISTRIBUTION)
            ctlv_expiracy_delta = np.random.choice(CTLV_OPTIONS, 1, p=CTLV_DISTRIBUTION)

            # Both directions for ease of access in send_packet
            self._latencies[edge_pair] = latency[0] if not test_mode else 0
            self._latencies[edge_pair[::-1]] = latency[0] if not test_mode else 0

            self._ctlv_deltas[edge_pair] = ctlv_expiracy_delta[0]
            self._ctlv_deltas[edge_pair[::-1]] = ctlv_expiracy_delta[0]

    def set_latency(self, a, b, latency):
        """Update the latency between two nodes. (assumes edge exists) """
        self._latencies[(a, b)] = latency
        self._latencies[(b, a)] = latency

    def get_ctlv_delta(self, a, b):
        return self._ctlv_deltas[(a, b)]

    def send_packet(self, src, dest, packet):
        """Processes and sends a packet from src to dest. """
        if self._debug: print(bcolours.HEADER + "[NODE-COMM] Relaying message between %s and %s." % (src, dest) + bcolours.ENDC)

        packet.set_timestamp(time.time() + self._latencies[(src, dest)])
        dest.receive_packet(packet, self._debug)

    def record_stat(self):
        """Placeholder method for recording stats - nodes can send when fail / success etc. """
        pass


def floor_msat(satoshi):
    """Floor to nearest millisatoshi. (lowest possible unit in LN) """
    return np.round(satoshi - 0.5 * 10**(-4), 4)

def calc_path_fees(G, path, amnt):
    """Calculate the compound path fees required for a given path.

    Note: compounding as amnt of equity moving per node is different!

    Args:
        G: NetworkX graph in use.
        path: (list<Node>) path for which fees are to be calculated.
        amnt: (float) number of satoshi to send to final Node in path.

    Returns:
        a list of satoshi to send at each hop.
    """
    hop_amnts = [amnt]
    for i in range(len(path)-2):
        fees = G[path[i+1]][path[i]]["fees"]  # Read fee rate from side nearest real source
        fee_this_hop = floor_msat(fees[0] + fees[1] * hop_amnts[-1]) # Fees always floored to nearest msat
        hop_amnts.append(fee_this_hop + hop_amnts[-1])
    return hop_amnts[::-1]

def calc_g_unbalance(G, pairs):
    """Calculate the unbalance of equity between payment channels.

    Higher equity channels take proportionally higher effect on the resultant ratio.

    Args:
        G: NetworkX graph in use.
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

    return str

def _dijkstra_reverse(G, source, target, initial_amnt, ignore_nodes=None, ignore_edges=None):
    """Find a path from source to target, starting the search from the target towards source.

    Only supports directed graph.

    Constaint: path must be able to support amnt with compounded fees.

    Uses Dijkstra's algorithm and the following is adapted from NetworkX src code.

    Args:
        G: NetworkX graph in use.
        source: (Node) source node.
        target: (Node) destination node.
        initial_amnt: (float) amnt of satoshi intended to be transferred on the last hop to the target.
        ignore_nodes: (set<Node>) nodes to ignore when path finding.
        ignore_edges: (set<Node>) edges to ignore when path finding.

    Returns:
        the length of the shortest path found, using the given weight function and,
        the corresponding shortest path found.
    """
    source, target = target, source  # Search from target instead

    # Weight - Fee calculation on directed edge for a given amnt
    weight = lambda d, amnt: d["fees"][0] + d["fees"][1] * amnt

    Gsucc = G.successors

    # Support optional nodes filter
    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w
            return iterate

        Gsucc = filter_iter(Gsucc)

    # Support optional edges filter
    if ignore_edges:
        def filter_succ_iter(succ_iter):
            def iterate(v):
                for w in succ_iter(v):
                    if (w, v) not in ignore_edges:  # Reversed as ignore_edges is real src to real dest
                        yield w
            return iterate

        Gsucc = filter_succ_iter(Gsucc)

    push = heappush
    pop = heappop
    dist = {}  # Dictionary of final distances
    seen = {}
    paths = {source: ([source], 0)}

    # Fringe is heapq with 3-tuples (distance, c, node)
    # Use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []

    seen[source] = 0
    push(fringe, (0, next(c), source))

    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # Already searched this node.
        dist[v] = d
        if v == target:
            break
        for w in Gsucc(v):
            # No fees on last hop
            cost = 0 if v == source else weight(G[w][v], paths[v][1] + initial_amnt)
            vw_dist = dist[v] + cost
            if w not in seen or vw_dist < seen[w]:
                # Add or update if new shortest distance from src -> u
                # But only if the path "supports" the amnt with fees
                # If this is the last hop, we know the equity at one side of the channel
                if (w == target and cost + initial_amnt <= G[w][v]["equity"]) or \
                    (w != target and cost + initial_amnt <= G[w][v]["equity"] + G[v][w]["equity"]):
                    seen[w] = vw_dist
                    push(fringe, (vw_dist, next(c), w))
                    paths[w] = (paths[v][0] + [w], cost)
    if target in dist:
        # Reverse the path!
        return dist[target], paths[target][0][::-1]
    else:
        # No path found
        return False


"""
//////////
SIMULATION
//////////
"""

def init_random_node(i):
    """Initialise and return a new node based on given simulation parameters, with given ID i."""
    merchant = np.random.choice([True, False], 1, p=[MERCHANT_PROB, 1 - MERCHANT_PROB])

    # Spend and receive freqs are a rating - for consisting in range [0, 1]
    # 0 - never selected, 1 - as likely as possible.
    spend_freq = 1
    receive_freq = 1

    return Node(i, merchant, spend_freq, receive_freq)

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
        the generated graph G,
        list of pairs of edges representing payment channels (one-direction, no duplicates).
    """
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.DiGraph()
    if test:
        nodes = [Node(i, False, 0, 0) for i in range(n)]
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
        node_comm: (NodeCommunicator) main communicator for each node to send packets to.
        seed: (int) seed for RNG. (None by default)

    Returns:
        the generated graph G,
        list of pairs of edges representing payment channels (one-direction, no duplicates).
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

def select_payment_nodes(consumers, merchants, total_freqs):
    """Select 2 nodes to attempt to pay from src to dest.

    Args:
        consumers: (list<Node>) consumer nodes within the graph.
        merchants: (list<Node>) merchant nodes within the graph.
        total_freq: (list) [sum of consumer spend freqs, sum of merchant spend freqs,
                            sum of consumer receive freqs, sum of merchant receive freqs]

    Returns:
        source and destination nodes to make the payment.
    """
    # Select src node - True denotes pick consumer
    src_consumer = np.random.choice([True, False], 1, p=[1 - ROLE_BIAS, ROLE_BIAS])
    srcs = consumers if src_consumer else merchants
    total_spend = total_freqs[0] if src_consumer else total_freqs[1]
    src_node = np.random.choice(srcs, 1, p=[src.get_spend_freq() / total_spend for src in srcs])[0]

    # Select dest node - True denotes pick merchant
    dest_merchant = np.random.choice([True, False], 1, p=[1 - ROLE_BIAS, ROLE_BIAS])
    dests = merchants if dest_merchant else consumers
    total_receive = total_freqs[3] if dest_merchant else total_freqs[2]
    dest_node = np.random.choice(dests, 1, p=[dest.get_receive_freq() / total_receive for dest in dests])[0]

    return src_node, dest_node

def select_pay_amnt(G, node):
    """Returns an appropriate spend amount (satoshi) for the selected node.

    Note: @TODO - do we select an always possible amount or allow for auto-failures?
    """
    max_amnt = node.get_largest_outgoing_equity(G)

    return np.random.randint(1, max_amnt + 1)

def simulator():
    """Main simulator calling function. """
    G, pairs = generate_wattz_strogatz(NUM_NODES, 6, 0.3)
    # G, pairs = generate_barabasi_albert(NUM_NODES, floor(NUM_NODES / 4))
    nodes = list(G)

    node_comm = NodeCommunicator(nodes, pairs, debug=False)

    consumers = []
    merchants = []
    total_freqs = [0, 0, 0, 0]  # s_c, s_m, r_c, r_m
    for node in nodes:
        if node.is_merchant():
            merchants.append(node)
            total_freqs[1] += node.get_spend_freq()
            total_freqs[3] += node.get_receive_freq()
        else:
            consumers.append(node)
            total_freqs[0] += node.get_spend_freq()
            total_freqs[2] += node.get_receive_freq()

    # Here - select nodes and attempt payments + record
    # Selected using select_payment_nodes
    # Need to decide how many satoshi per payment (based on available balance etc?)
    # Need to decide how to split up payments - trial & error.
    # Question: do we avoid selecting src_nodes that can't send money or count that as failure?

    if DEBUG: print("// INIT NUM_STARTING_PAYMENTS PAYMENTS\n-------------------------")

    for i in range(NUM_STARTING_PAYMENTS):
        selected = select_payment_nodes(consumers, merchants, total_freqs)
        amnt = select_pay_amnt(G, selected[0])

        selected[0].init_payment(G, selected[1], amnt, node_comm)

    if DEBUG: print("// SIMULATION LOOP BEGIN\n-------------------------")

    total_payments_sent = NUM_STARTING_PAYMENTS
    while success_count + fail_count < total_payments_sent:
        for node in nodes:  # For now, pop one packet per node
            node.process_next_packet(G, node_comm)

    print("// ANALYTICS\n-------------------------")
    print("# successful payments:", success_count)
    print("# failed payments:", fail_count)
    print("# amp retries:", amp_retry_count)

    # nx.draw(G)
    # plt.show()

"""
//////////////////////////
CORE FUNCTIONALITY TESTING
//////////////////////////
"""

def test_func():
    TEST_DEBUG = False

    ################## func: _dijkstra_reverse ###########
    if TEST_DEBUG: print("// PATH FINDING\n-------------------------")

    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    node_comm = NodeCommunicator(nodes, pairs, test_mode=True, debug=TEST_DEBUG)

    length, path = _dijkstra_reverse(G, nodes[0], nodes[14], 5)

    assert path == [nodes[0], nodes[19], nodes[14]]

    ################## func: send_direct ###########
    if TEST_DEBUG: print("// SUCCESSFUL DIRECT TRANSFER\n-------------------------")

    e_4_5 = G[nodes[4]][nodes[5]]["equity"]
    e_5_4 = G[nodes[5]][nodes[4]]["equity"]

    nodes[4].send_direct(G, nodes[5], 10, node_comm, debug=TEST_DEBUG)
    nodes[5].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    assert G[nodes[4]][nodes[5]]["equity"] == e_4_5 - 10
    assert G[nodes[5]][nodes[4]]["equity"] == e_5_4 + 10

    if TEST_DEBUG: print("// FAILED DIRECT TRANSFER - NO CHANNEL\n-------------------------")

    attempt = nodes[4].send_direct(G, nodes[17], 5, node_comm, debug=TEST_DEBUG)
    assert attempt == False

    ################## func: _init_payment [NON-AMP] ###########
    if TEST_DEBUG: print("// SUCCESSFUL PAYMENT\n-------------------------")

    # Payment from 0 to 14 will find path 0-19-14
    path = [nodes[0], nodes[1], nodes[14]]
    amnts = calc_path_fees(G, path, 5)

    e_0_19 = G[nodes[0]][nodes[19]]["equity"]
    e_19_0 = G[nodes[19]][nodes[0]]["equity"]
    e_19_14 = G[nodes[19]][nodes[14]]["equity"]
    e_14_19 = G[nodes[14]][nodes[19]]["equity"]

    # Initialise by sending message from 0 to 1 and propagating out and in.
    nodes[0]._init_payment(G, nodes[14], 5, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - amnts[0]
    assert G[nodes[19]][nodes[0]]["equity"] == e_19_0 + amnts[0]
    assert G[nodes[19]][nodes[14]]["equity"] == e_19_14 - amnts[1]
    assert G[nodes[14]][nodes[19]]["equity"] == e_14_19 + amnts[1]

    if TEST_DEBUG: print("\n// FAILED PAYMENT\n-------------------------")

    # Equity from 19 to 14 depleted too much, so try this route again to test
    # for funds being released back correctly when a route fails mid-way forward.
    nodes[0]._init_payment(G, nodes[14], 25, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    # Should be the same as payment only made it one hop.
    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - amnts[0]
    assert G[nodes[19]][nodes[0]]["equity"] == e_19_0 + amnts[0]
    assert G[nodes[19]][nodes[14]]["equity"] == e_19_14 - amnts[1]
    assert G[nodes[14]][nodes[19]]["equity"] == e_14_19 + amnts[1]

    # No route available
    if TEST_DEBUG: print("\n// NO ROUTE AVAILABLE\n-------------------------")
    assert nodes[0]._init_payment(G, nodes[14], 100, node_comm, test_mode=True, debug=TEST_DEBUG) == False

    # One-hop payments - make sure finds direct path.
    if TEST_DEBUG: print("\n// ONE-HOP PAYMENT\n-------------------------")

    path = [nodes[11], nodes[9]]
    amnts = calc_path_fees(G, path, 5)

    e_11_9 = G[nodes[11]][nodes[9]]["equity"]
    e_9_11 = G[nodes[9]][nodes[11]]["equity"]

    nodes[11]._init_payment(G, nodes[9], 5, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[9].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    assert G[nodes[11]][nodes[9]]["equity"] == e_11_9 - amnts[0]
    assert G[nodes[9]][nodes[11]]["equity"] == e_9_11 + amnts[0]

    ################## func: make_payment [AMP] ################
    # Re-init just in case
    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    # Need to re-init for new node/edge references
    node_comm = NodeCommunicator(nodes, pairs, test_mode=True, debug=TEST_DEBUG)

    if TEST_DEBUG: print("\n// AMP SUCCESS - NO FAILED ROUTES\n-------------------------")

    # From above, we've already tested that single payments work and if they fail revert correctly
    # So, for AMP - need to check:
    #   1) split payment that works straight off
    #   2) fails overall with subpayments that didn't fail reverting
    #   3) fails, retries routes and works

    # Works with no fails - right now, this means it tries same route and keeps working.
    # Again 0-19-14 sent as packets of 3/3/4
    path = [nodes[0], nodes[19], nodes[14]]
    amnts = calc_path_fees(G, path, 3)
    amnts_2 = calc_path_fees(G, path, 4)

    e_0_1 = G[nodes[0]][nodes[1]]["equity"]
    e_1_0 = G[nodes[1]][nodes[0]]["equity"]
    e_1_14 = G[nodes[1]][nodes[14]]["equity"]
    e_14_1 = G[nodes[14]][nodes[1]]["equity"]
    e_0_19 = G[nodes[0]][nodes[19]]["equity"]
    e_19_0 = G[nodes[19]][nodes[0]]["equity"]
    e_19_14 = G[nodes[19]][nodes[14]]["equity"]
    e_14_19 = G[nodes[14]][nodes[19]]["equity"]

    nodes[0].init_payment(G, nodes[14], 10, node_comm, k=3, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - amnts[0] * 2
    assert G[nodes[19]][nodes[0]]["equity"] - (e_19_0 + amnts[0] * 2) <= 0.01  # Rounding issue
    assert G[nodes[19]][nodes[14]]["equity"] == e_19_14 - amnts[1] * 2
    assert G[nodes[14]][nodes[19]]["equity"] == e_14_19 + amnts[1] * 2
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts_2[0]
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + amnts_2[0]
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts_2[1]
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1 + amnts_2[1]

    if TEST_DEBUG: print("\n// AMP SUCCESS - FAILS: REVERTED PARTIAL PAYMENTS\n-------------------------")

    e_0_1 = G[nodes[0]][nodes[1]]["equity"]
    e_1_0 = G[nodes[1]][nodes[0]]["equity"]
    e_1_14 = G[nodes[1]][nodes[14]]["equity"]
    e_14_1 = G[nodes[14]][nodes[1]]["equity"]
    e_0_19 = G[nodes[0]][nodes[19]]["equity"]
    e_19_0 = G[nodes[19]][nodes[0]]["equity"]
    e_19_14 = G[nodes[19]][nodes[14]]["equity"]
    e_14_19 = G[nodes[14]][nodes[19]]["equity"]

    # Try same payment again - will work for subpayments but middle payment will fail.
    # First and last subpayments reverted - so should be same as above.
    # Stops when all routes exhausted - will also stop if taking too long - no need to test this.
    nodes[0].init_payment(G, nodes[14], 10, node_comm, k=3, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    # 0-19-14 will right now have 2 partial payments locked up in HTLC
    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - amnts[0]
    assert G[nodes[19]][nodes[0]]["equity"] == e_19_0  # Same as no preimage received
    assert G[nodes[19]][nodes[14]]["equity"] == e_19_14 - amnts[1]
    assert G[nodes[14]][nodes[19]]["equity"] == e_14_19
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts_2[0]
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0  # Same as no preimage received
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts_2[1]
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1

    # Release 2 subpayments that made the distance.
    nodes[14].clear_old_amps(G, node_comm, True, TEST_DEBUG)

    # Pretend all routes exhausting here rather than physically trying them.
    # Because all routes aren't actually exhausted, when a subpayment comes back - keeps resending.

    if TEST_DEBUG: print("\n// AMP SUCCESS - SUCCESS WITH FAILED ROUTES\n-------------------------")

    # Try a payment that has failed routes, but finds enough other routes to
    # successfully route the payment - reset so we can use same route.
    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    # Need to re-init for new node/edge references
    node_comm = NodeCommunicator(nodes, pairs, test_mode=True, debug=TEST_DEBUG)

    # Sent as 5/5/5 - alternative route 0-1-14 for last two payments.
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

    nodes[0].init_payment(G, nodes[14], 15, node_comm, k=3, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    # New route 0-1-14
    nodes[1].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[14].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    # Knows base preimage, so HTLCs released back
    nodes[19].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].process_next_packet(G, node_comm, test_mode=True, debug=TEST_DEBUG)

    # 2 on 0-1-14, 1 on 0-19-14
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - amnts[0] * 2
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + amnts[0] * 2
    assert G[nodes[1]][nodes[14]]["equity"] == e_1_14 - amnts[1] * 2
    assert G[nodes[14]][nodes[1]]["equity"] == e_14_1 + amnts[1] * 2
    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - amnts_2[0]
    assert G[nodes[19]][nodes[0]]["equity"] == e_19_0 + amnts_2[0]
    assert G[nodes[19]][nodes[14]]["equity"] == e_19_14 - amnts_2[1]
    assert G[nodes[14]][nodes[19]]["equity"] == e_14_19 + amnts_2[1]

    if TEST_DEBUG: print("\n// PACKET ARRIVAL TIME PROCESSING - FUTURE PACKETS\n-------------------------")

    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    # Need to re-init for new node/edge references
    node_comm = NodeCommunicator(nodes, pairs, test_mode=True, debug=TEST_DEBUG)

    e_0_1 = G[nodes[0]][nodes[1]]["equity"]
    e_1_0 = G[nodes[1]][nodes[0]]["equity"]

    node_comm.set_latency(nodes[0], nodes[1], 10)
    nodes[0]._init_payment(G, nodes[1], 5, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, debug=TEST_DEBUG)

    # Should be no change as packet hasn't "reached" 1 yet.
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0

    # Try again with no latency, new payment, old still won't have made it
    node_comm.set_latency(nodes[0], nodes[1], 0)
    nodes[0]._init_payment(G, nodes[1], 5, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[1].process_next_packet(G, node_comm, debug=TEST_DEBUG)

    # Should be no change as packet hasn't "reached" 1 yet.
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - 5
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + 5

simulator()

if __name__ == "__main__":
    test_func()
