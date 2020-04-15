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

DEBUG = False
NUM_NODES = 2000
NUM_TEST_NODES = 20
TOTAL_TO_SEND = 7000

AMP_RESEND = False  # Retry flag for AMP payments
JIT_ROUTING = False
JIT_RESERVE = False
JIT_REPEAT_REBALANCE = True
JIT_FULL_KNOWLEDGE = False

FEE_BALANCING = False
FEE_BALANCING_CHECKPOINT = 5
FEE_BALANCING_UPDATE_POINT = 0.2

MAX_HOPS = 20
AMP_TIMEOUT = 60
MERCHANT_PROB = 0.67
LATENCY_OPTIONS = [0.1, 1, 10]  # Sprites paper, in seconds
LATENCY_DISTRIBUTION = [0.925, 0.049, 0.026]
CTLV_OPTIONS = [9]  # Default on LN for now
CTLV_DISTRIBUTION = [1]
CLTV_MULTIPLIER = 600  # BTC block time is ~10 minutes, so 600 seconds
DELAY_RESP_VAL = [0, 0.2, 0.4, 0.6, 0.8, 1]  # Amount of timelock time node will wait before cancelling/claiming
DELAY_RESP_DISTRIBUTION = [1, 0, 0, 0, 0, 0]#0.9, 0.05, 0.01, 0.005, 0.005, 0.03]
MAX_PATH_ATTEMPTS = 30  # For AMP, if we try a subpayment on this many paths and fail, stop
ENCRYPTION_DELAYS = [0]  # For now, setting these to 0 as latency should include this.
ENCRYPTION_DELAY_DISTRIBUTION = [1]
DECRYPTION_DELAYS = [0]
DECRYPTION_DELAY_DISTRIBUTION = [1]

# Consumers pay merchants - this is the chance for different role,
# i.e. merchant to pay or consumer to receive, [0, 0.5] - 0.5 means both as likely as each other
ROLE_BIAS = 0.0
HIGH_FUNDS_CHANCE = 0.2

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
total_path_hops = 0
total_paths_found = 0
path_too_large_count = 0

# Packet handling
sim_time = 0.0
packet_queue = PriorityQueue()

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
        jit_id: (UUID) if involved in JIT rebalancing, store the ID.
    """

    def __init__(self, path, index_mapping, send_amnts, type="pay_htlc", subpayment=False, jit_id=False):
        self._path = path
        self._index_mapping = index_mapping
        self._type = type
        self._send_amnts = send_amnts
        self._subpayment = subpayment
        self._jit_id = jit_id
        self._htlc_ids = []
        self._timestamp = None
        self._final_index = len(path) - 1
        self._delayed = [False] * (len(path) - 1)

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

    def get_jit_id(self):
        return self._jit_id

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

    def is_delayed(self, index):
        return self._delayed[index]

    def set_delayed(self, index):
        self._delayed[index] = True

    def __lt__(self, other):
        return self.get_timestamp() < other.get_timestamp()


class Node:
    """Singular lightning node.

    Attributes:
        id: (int) ID of node.
        merchant: (boolean) True if a merchant, False if consumer. (consumer more likely to spend)
        spend_freq: (float) likelihood of selection to make next payment.
        receive_freq: (float) likelihood of selection to receive next payment.
        encryption_delay: (float) seconds it takes to encrypt a full packet.
        decryption_delay: (float) seconds it takes to decrypt a layer of a packet.
    """

    def __init__(self, id, merchant, spend_freq, receive_freq, encryption_delay, decryption_delay):
        self._id = id
        self._merchant = merchant
        self._spend_freq = spend_freq
        self._receive_freq = receive_freq
        self._encryption_delay = encryption_delay
        self._decryption_delay = decryption_delay
        self._amp_out = {}  # Outgoing AMPs still active
        self._amp_in = {}  # Awaiting incoming AMPs
        self._htlc_out = {}  # Maps other nodes to HTLCs
        self._jit_out = {}  # Maps JIT route IDs to payment packets (to be completed after if JIT successful)
        self._jit_reserve = {}  # Total number of equity reserved while rebalancing, mapped by out-going edges

    def get_id(self):
        return self._id

    def is_merchant(self):
        return self._merchant

    def get_spend_freq(self):
        return self._spend_freq

    def get_receive_freq(self):
        return self._receive_freq

    def get_decryption_delay(self):
        return self._decryption_delay

    def get_jit_reserve(self, node=None):
        if node is None:
            return self._jit_reserve
        return self._jit_reserve[node]

    def get_htlc_out(self, node):
        """Get HTLCs locked up with a specific node."""
        if node in self._htlc_out:
            return self._htlc_out[node]
        else:
            return None

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

    def receive_packet(self, packet, debug=DEBUG):
        """Receive a packet from the main node communicator.

        Args:
            packet: (Packet) the packet to add to the Node's priority queue.
            debug: (boolean) True to display debug print statements.
        """
        global packet_queue

        if debug: print(bcolours.OKGREEN + "%s is receiving packet from node communicator." % self + bcolours.ENDC)
        packet_queue.put((packet, self))

    def process_packet(self, G, node_comm, p, test_mode=False, amp_resend_enabled=AMP_RESEND, jit_enabled=JIT_ROUTING, debug=DEBUG):
        """Process next packet if any.

        Args:
            G: NetworkX graph in use.
            debug: (boolean) True to display debug print statements.
            node_comm: (NodeCommunicator) Main message communicator between nodes.
            p: (Packet) Packet to process.
            test_mode: (boolean) True if running from test function.
            debug: (boolean) True to display debug print statements.
        """
        # Analytics
        global success_count
        global fail_count
        global amp_retry_count

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

            jit_reserve = 0
            if self in src.get_jit_reserve():
                jit_reserve = src.get_jit_reserve(self)

            # Assume: amnt > 0, check for available funds only
            if G[src][self]["equity"] - jit_reserve >= amnt:
                G[src][self]["equity"] -= amnt  # To HTLC (if not direct) or other node (if direct)
                if type == "pay":
                    if not subpayment:
                        success_count += 1
                        G[self][src]["equity"] += amnt
                    else:
                        id, subp_index, _, ttl = p.get_subpayment()
                        if not id in self._amp_in:
                            self._amp_in[id] = []

                        if sim_time <= ttl:  # Within time
                            self._amp_in[id].append(p)
                            if debug: print(bcolours.OKGREEN + "Received partial payment at %s. [%d]" % (self, subpayment[1]) + bcolours.ENDC)

                            if len(self._amp_in[id]) == subpayment[2]:  # All collected, release HTLCs
                                for s in self._amp_in[id]:
                                    index = s.get_index(self._id)
                                    dest = s.get_node(index - 1)
                                    amnt = s.get_amnt(index - 1)
                                    G[self][dest]["equity"] += amnt

                                    # Even if pay and not pay_htlc - can still send preimage, won't change anything.
                                    s.set_type("preimage")
                                    s.set_timestamp(sim_time + dest.get_decryption_delay())

                                    if debug: print("Sending preimage release message to node communicator from %s (-> %s). [%d]" % (self, dest, subpayment[1]))
                                    node_comm.send_packet(self, dest, s)
                                del self._amp_in[id]
                        else:  # Out of time, need to release subpayments back
                            for s in self._amp_in[id]:
                                index = s.get_index(self._id)
                                dest = s.get_node(index - 1)

                                s.set_type("cancel")
                                s.set_timestamp(sim_time + dest.get_decryption_delay())

                                if debug: print("Sending cancel message to node communicator from %s (-> %s). [%d]" % (self, dest, subpayment[1]))
                                node_comm.send_packet(self, dest, s)
                            del self._amp_in[id]
                    if debug: print("Sent %.4f from %s to %s." % (amnt, src, self) + (" [%d]" % subpayment[1] if subpayment else ""))
                else:
                    id = uuid4()
                    ttl = sim_time + node_comm.get_ctlv_delta(src, self) * CLTV_MULTIPLIER
                    src.update_htlc_out(self, id, amnt, ttl)

                    p.add_htlc_id(id)

                    if debug: print("Sent (HTLC) %.4f from %s to %s." % (amnt, src, self) + (" [%d]" % subpayment[1] if subpayment else ""))
                    if index == p.get_final_index():
                        # Receiver - so release preimage back and claim funds if not AMP
                        if not subpayment:
                            G[self][src]["equity"] += amnt

                            dest = p.get_node(index - 1)
                            p.set_type("preimage")
                            # p.set_timestamp(sim_time + dest.get_decryption_delay())

                            if debug: print("Received payment - sending preimage release message to node communicator from %s (-> %s)." % (self, dest))
                            node_comm.send_packet(self, dest, p)

                            # Clear reverse
                            if JIT_RESERVE:
                                original_amnt = original_p.get_amnt(index)
                                self._jit_reserve[dest] = 0

                            jit_id = p.get_jit_id()
                            if jit_id: # Now (might be) enough funds are available after rebalance.
                                original_p = self._jit_out[jit_id]
                                del self._jit_out[jit_id]

                                index = original_p.get_index(self._id)
                                dest = original_p.get_node(index + 1)
                                amnt = original_p.get_amnt(index)

                                # But if another payment stole the funds while rebalancing, try to rebalance again
                                if JIT_REPEAT_REBALANCE and G[self][dest]["equity"] < amnt:
                                    found_jit_route = False
                                    if jit_enabled and not original_p.get_jit_id():  # Don't create rebalance loops
                                        rebalance_delta = amnt - G[self][dest]["equity"]
                                        cost, path = _jit_dijsktra_reverse(G, self, dest, rebalance_delta)

                                        if len(path) == 4:
                                            found_jit_route = True
                                            jit_id = uuid4()
                                            self._jit_out[jit_id] = original_p  # @TODO: decryption delay here too

                                            # Reserve equity, so another payment doesn't take it while rebalancing
                                            if JIT_RESERVE:
                                                if dest not in self._jit_reserve:
                                                    self._jit_reserve[dest] = 0

                                                self._jit_reserve[dest] += G[self][dest]["equity"]

                                            send_amnts = calc_path_fees(G, path, rebalance_delta)
                                            index_mapping = {path[i].get_id(): i for i in range(len(path))}
                                            index_mapping["src"] = self

                                            if debug: print("Attempting to rebalancing funds by sending from %s to %s for JIT payment." % (self, path[1]))

                                            jit_p = Packet(path, index_mapping, send_amnts, jit_id=jit_id)
                                            jit_p.set_timestamp(sim_time)
                                            # jit_p.set_timestamp(sim_time + self._encryption_delay + path[1].get_decryption_delay())

                                            if debug: print("Sending %s message to node communicator from %s (-> %s)." % (type, self, path[1]) + (" [%d]" % subpayment[1] if subpayment else ""))
                                            node_comm.send_packet(self, path[1], jit_p)

                                    if not found_jit_route:
                                        if debug: print(bcolours.FAIL + "Error: equity between %s and %s not available for transfer - reversing." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
                                        original_p.set_type("cancel")
                                        dest = original_p.get_node(index - 1)
                                        original_p.set_timestamp(sim_time + dest.get_decryption_delay())

                                        node_comm.send_packet(self, dest, original_p)
                                else:
                                    original_p.set_timestamp(sim_time + dest.get_decryption_delay())

                                    if debug: print("Sending pay_htlc message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                                    node_comm.send_packet(self, dest, original_p)
                        else:  # If AMP, we need to keep and store all the partial payments
                            id, subp_index, _, ttl = p.get_subpayment()
                            if not id in self._amp_in:
                                self._amp_in[id] = []

                            if sim_time <= ttl:  # Within time
                                self._amp_in[id].append(p)
                                if debug: print(bcolours.OKGREEN + "Received partial payment at %s. [%d]" % (self, subpayment[1]) + bcolours.ENDC)

                                if len(self._amp_in[id]) == subpayment[2]:  # All collected, release HTLCs
                                    for s in self._amp_in[id]:
                                        index = s.get_index(self._id)
                                        dest = s.get_node(index - 1)
                                        amnt = s.get_amnt(index - 1)
                                        G[self][dest]["equity"] += amnt

                                        s.set_type("preimage")
                                        s.set_timestamp(sim_time + dest.get_decryption_delay())

                                        if debug: print("Sending preimage release message to node communicator from %s (-> %s). [%d]" % (self, dest, subpayment[1]))
                                        node_comm.send_packet(self, dest, s)
                                    del self._amp_in[id]
                            else:  # Out of time, need to release subpayments back
                                for s in self._amp_in[id]:
                                    index = s.get_index(self._id)
                                    dest = s.get_node(index - 1)

                                    s.set_type("cancel")
                                    s.set_timestamp(sim_time + dest.get_decryption_delay())

                                    if debug: print("Sending cancel message to node communicator from %s (-> %s). [%d]" % (self, dest, subpayment[1]))
                                    node_comm.send_packet(self, dest, s)
                                del self._amp_in[id]
                    else:
                        # Need to keep sending it on, but only if funds are available
                        dest = p.get_node(index + 1)
                        amnt = p.get_amnt(index)

                        jit_reserve = 0
                        if dest in self.get_jit_reserve():
                            jit_reserve = self.get_jit_reserve(dest)

                        if G[self][dest]["equity"] - jit_reserve >= amnt:
                            p.set_timestamp(sim_time + dest.get_decryption_delay())

                            if debug: print("Sending pay_htlc message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                            node_comm.send_packet(self, dest, p)

                            G[self][dest]["fee_metrics"][1] += 1  # Sent payment on down this edge, record
                        else:
                            # If JIT routing is turned on, try to rebalance.
                            found_jit_route = False
                            if jit_enabled and not p.get_jit_id():  # Don't create rebalance loops
                                rebalance_delta = amnt - G[self][dest]["equity"]
                                cost, path = _jit_dijsktra_reverse(G, self, dest, rebalance_delta)

                                if len(path) == 4:  # Proper cycle found
                                    found_jit_route = True
                                    jit_id = uuid4()
                                    self._jit_out[jit_id] = p  # @TODO: decryption delay here too

                                    # Reserve equity, so another payment doesn't take it while rebalancing
                                    if JIT_RESERVE:
                                        if dest not in self._jit_reserve:
                                            self._jit_reserve[dest] = 0

                                        self._jit_reserve[dest] += G[self][dest]["equity"]

                                    send_amnts = calc_path_fees(G, path, rebalance_delta)
                                    index_mapping = {path[i].get_id(): i for i in range(len(path))}
                                    index_mapping["src"] = self

                                    if debug: print("Attempting to rebalancing funds by sending from %s to %s for JIT payment." % (self, path[1]))

                                    jit_p = Packet(path, index_mapping, send_amnts, jit_id=jit_id)
                                    jit_p.set_timestamp(sim_time + self._encryption_delay + path[1].get_decryption_delay())

                                    if debug: print("Sending %s message to node communicator from %s (-> %s)." % (type, self, path[1]) + (" [%d]" % subpayment[1] if subpayment else ""))
                                    node_comm.send_packet(self, path[1], jit_p)

                            if not found_jit_route:
                                if debug: print(bcolours.FAIL + "Error: equity between %s and %s not available for transfer - reversing." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
                                p.set_type("cancel")
                                p.set_timestamp(sim_time + p.get_node(index - 1).get_decryption_delay())

                                node_comm.send_packet(self, p.get_node(index - 1), p)
            else:
                if debug: print(bcolours.FAIL + "Error: equity between %s and %s not available for transfer." % (src, self) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)

                dest = p.get_node(index - 1)  # Propagate back CANCEL
                p.set_type("cancel_rest")  # Not regular cancel as no HTLC with prev. node.
                p.set_timestamp(sim_time + dest.get_decryption_delay())

                if debug: print("Sending cancellation message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                node_comm.send_packet(self, dest, p)
        elif type == "preimage":
            # Release HTLC entry
            if p.get_index("src") == self:
                del self._htlc_out[p.get_node(1)][p.get_htlc_id(0)]
            else:
                if p.get_htlc_id(index) in self._htlc_out[p.get_node(index + 1)]:
                    del self._htlc_out[p.get_node(index + 1)][p.get_htlc_id(index)]

            if index != 0 and p.get_index("src") != self:  # Keep releasing
                dest = p.get_node(index - 1)
                G[self][dest]["equity"] += amnt

                if debug:
                    print("%s claimed back %.4f from payment from %s." % (self, amnt, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                    print("Sending preimage release message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))

                p.set_timestamp(sim_time + dest.get_decryption_delay())
                node_comm.send_packet(self, dest, p)
            else:  # Sender, receipt come back
                if not subpayment and not p.get_jit_id():
                    success_count += 1

                if debug:
                    j = p.get_final_index()
                    print(bcolours.OKGREEN + "Payment [%s -> %s // %.4f] successfully completed." % (self, p.get_node(j), p.get_amnt(j-1)) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)

                if p.get_subpayment() and p.get_subpayment()[0] in self._amp_out:
                    # Then, receiver must have got all of AMP
                    del self._amp_out[p.get_subpayment()[0]]
                    success_count += 1

                    j = p.get_final_index()
                    if debug: print(bcolours.OKGREEN + "AMP from %s to %s [%.4s] fully completed." % (self, p.get_node(j), p.get_amnt(j-1)) + bcolours.ENDC)
        else:  # Cancel message
            if p.get_index("src") == self:
                index = 0

            dest = p.get_node(index + 1)
            amnt = p.get_amnt(index)

            if not p.is_delayed(index):
                cancel_chance = np.random.choice(DELAY_RESP_VAL, 1, p=DELAY_RESP_DISTRIBUTION)[0] if not test_mode else 0

                # Some unresponsive or malicious nodes might delay the cancellation
                if cancel_chance:
                    # Simulate this by requeuing the Packet with an updated timestamp to open
                    # This isn't entirely correct - just does period of delta - possibly much larger... @TODO
                    ttl = sim_time + node_comm.get_ctlv_delta(self, dest) * CLTV_MULTIPLIER * cancel_chance
                    p.set_timestamp(ttl)
                    p.set_delayed(index)
                    self._queue.put(p)

                    if debug: print("%s is waiting for %.4f of HTLC timeout before signing..." % (self, cancel_chance))
                    return

            if p.get_type() == "cancel":  # Not cancel_rest
                if p.get_htlc_id(index) in self._htlc_out[dest]:
                    del self._htlc_out[dest][p.get_htlc_id(index)]

                G[self][dest]["equity"] += amnt  # Claim back
                if debug: print("%s cancelled HTLC and claimed back %.4f from payment to %s." % (self, amnt, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
            else:
                p.set_type("cancel")

            if index == 0:
                if debug:
                    j = p.get_final_index()
                    print(bcolours.FAIL + "Payment [%s -> %s // %.4f] failed and returned." % (self, p.get_node(j), p.get_amnt(j-1)) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
                jit_id = p.get_jit_id()
                if jit_id:
                    original_p = self._jit_out[jit_id]
                    del self._jit_out[jit_id]

                    original_index = original_p.get_index(self._id)
                    original_dest = original_p.get_node(original_index - 1)

                    # Clear reserve
                    if JIT_RESERVE:
                        original_out_amnt = original_p.get_amnt(original_index)
                        self._jit_reserve[original_p.get_node(original_index + 1)] = 0

                    original_p.set_type("cancel")
                    original_p.set_timestamp(sim_time + original_dest.get_decryption_delay())

                    if debug: print("Sending cancel message to node communicator from %s (-> %s)." % (self, original_dest))
                    node_comm.send_packet(self, original_dest, original_p)

            if index != 0:  # Send cancel message on
                G[self][dest]["fee_metrics"][2] += 1  # Record fail coming back from payment relayed on

                # Check metrics and adjust fee rate accordingly
                if FEE_BALANCING and G[self][dest]["fee_metrics"][1] > FEE_BALANCING_CHECKPOINT:
                    fail_return = G[self][dest]["fee_metrics"][2] / G[self][dest]["fee_metrics"][1]
                    old = G[self][dest]["fee_metrics"][0]
                    if old is not None:
                        diff = fail_return - old
                        if diff < -FEE_BALANCING_UPDATE_POINT:  # Too high, increase
                            # print("REDUCING")
                            fees = G[self][dest]["fees"]
                            if fees[0] > 0.0: fees[0] -= 0.1
                            if fees[1] > 0.0: fees[1] -= 0.0000001
                        elif diff > FEE_BALANCING_UPDATE_POINT:
                            # print("INCREASING")
                            fees = G[self][dest]["fees"]
                            fees[0] += 0.1
                            fees[1] += 0.0000001
                    G[self][dest]["fee_metrics"] = [fail_return, 0, 0]  # Re-init

                dest = p.get_node(index - 1)
                p.set_timestamp(sim_time + dest.get_decryption_delay())

                if debug: print("Sending cancellation message to node communicator from %s (-> %s)." % (self, dest) + (" [%d]" % subpayment[1] if subpayment else ""))
                node_comm.send_packet(self, dest, p)
            elif subpayment:  # Partial payment of AMP failed
                # subpayment in form [ID, index, total num, ttl]
                id, i, k, ttl = subpayment

                if id in self._amp_out:  # Still on-going
                    j = p.get_final_index()
                    if amp_resend_enabled and sim_time < ttl and self._amp_out[id][2]:
                        # Still within time limit - so try again!
                        # In Rusty Russell podcast - c-lightning is 60 seconds.
                        self._amp_out[id][0][i].append(p.get_path())
                        new_subp = (id, i, k)

                        if debug: print(bcolours.WARNING + "Resending... [%d]" % i + bcolours.ENDC)
                        self._init_payment(G, p.get_node(j), p.get_amnt(j-1), node_comm, new_subp, test_mode=test_mode, debug=debug)

                        amp_retry_count += 1
                    else:
                        fail_count += 1
                        del self._amp_out[id]

                        # Clear stored packets
                        p.get_node(j).clear_old_amps(G, node_comm)

                        if debug:
                            j = p.get_final_index()
                            print(bcolours.FAIL + "Partial payment [%d] of failed AMP from %s to %s [%.4s] returned - not resending." % (i, self, p.get_node(j), p.get_amnt(j-1)) + bcolours.ENDC)
                else:
                    if debug:
                        j = p.get_final_index()
                        print(bcolours.FAIL + "Partial payment [%d] of failed AMP from %s to %s [%.4s] returned - not resending." % (i, self, p.get_node(j), p.get_amnt(j-1)) + bcolours.ENDC)
            else:  # Non-subpayment failed
                if not p.get_jit_id(): fail_count += 1

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
            if force or (len(self._amp_in[id]) and self._amp_in[id][0].get_subpayment()[3] - sim_time < 0):
                if debug: print(bcolours.WARNING + "Cancelling partial subpayments from %s for AMP ID %s" % (self, id) + bcolours.ENDC)
                for p in self._amp_in[id]:
                    index = p.get_index(self._id)
                    dest = p.get_node(index - 1)
                    amnt = p.get_amnt(index - 1)

                    p.set_type("cancel")
                    p.set_timestamp(sim_time + dest.get_decryption_delay())

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
            index_mapping["src"] = self

            p = Packet(path, index_mapping, [amnt], "pay", False)
            p.set_timestamp(sim_time + dest.get_decryption_delay())
            node_comm.send_packet(self, dest, p)

            if debug: print("Sending pay message to node communicator from %s (-> %s)." % (self, dest))
            return True
        else:
            if debug: print(bcolours.FAIL + "Error: no direct payment channel between %s and %s." % (self, dest) + bcolours.ENDC)
            return False

    def _find_path(self, G, dest, amnt, failed_paths, k=False):
        """Attempt to find the next shortest path from self to dest within graph G, that is not in failed_paths.

        Adapted from NetworkX shorest_simple_paths src code.

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to find best path to.
            amnt: (float) number of satoshi the resultant path must support.
            failed_paths: (list) previously yielded routes to skip.
            k: (int) number of new paths to find.

        Returns:
            a generator that produces a list of possible paths, from best to worst,
            or False if no paths exist.

        Raises:
            NetworkXError: if self or dest are not in the input graph.
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

        num_to_find = len(failed_paths) if not k else k - 1

        # Find one new path per func call, up until a global maximum.
        while len(listA) <= num_to_find and len(listA) < MAX_PATH_ATTEMPTS:
            avoid_edges = []
            to_avoid = listA if k else failed_paths
            for p in to_avoid:
                for i in range(len(p) - 1):
                    avoid_edges.append((p[i], p[i+1]))

            attempt = shortest_path_func(G, self, dest, amnt, avoid_edges)

            if attempt:
                length, path = attempt
                listB.push(length, path)

            if listB:
                path = listB.pop()
                yield path
                listA.append(path)
            else:
                return False

    def _init_payment(self, G, dest, amnt, node_comm, subpayment=False, routes=[], route_index=0, test_mode=False, debug=DEBUG):
        """Initialise a regular single payment from this node to destination node of amnt.

        Args:
            G: NetworkX graph in use.
            dest: (Node) destination node to attempt to send money to.
            amnt: (float) number of satoshi to send.
            node_comm: (NodeCommunicator) main communicator to send packets to.
            subpayment: (boolean / (UUID, int, int))
                - (ID of subpayment group, index of subpayment, total num subpayments),
                - or False if not AMP.
            routes: (List<List<Node>>) preset routes to try.
            route_index: (int) index for which route in routes to try for this subpayment.
            test_mode: (boolean) True if running from test function.
            debug: (boolean) True to display debug print statements.

        Returns:
            True if payment packet sent out successfully,
            False otherwise.
        """
        # Analytics
        global fail_count
        global total_path_hops
        global total_paths_found
        global path_too_large_count

        if subpayment:
            if len(routes) > 0:
                paths = [routes[route_index]]

                # For now, set all failed_routes for future as routes
                id, i, n = subpayment
                self._amp_out[id][0][i] = routes
            else:  # Won't go in here for my experiments
                id, i, n = subpayment
                failed_routes = self._amp_out[id][0][i]

                paths = [p for p in self._find_path(G, dest, amnt, failed_routes)]
        else:
            # Only try best path once for regular payments
            paths = [_slack_based_reverse(G, self, dest, amnt)]

        # For my experiments with AMP, clause won't be entered
        if len(paths) == 0 or paths[0] is False:
            if debug: print(bcolours.FAIL + "Error: no possible routes available." + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
            if subpayment:
                self._amp_out[id][2] = False
                if debug: print(bcolours.FAIL + "AMP from %s to %s [%.4s] failed." % (self, dest, amnt) + bcolours.ENDC)
            else:
                fail_count += 1
            return False

        path = paths[0] if subpayment else paths[0][1]
        total_path_hops += len(path)
        total_paths_found += 1

        if len(path) - 1 > MAX_HOPS:
            path_too_large_count += 1

            if debug: print(bcolours.FAIL + "Error: path exceeds max-hop distance. (%d)" % (len(path) - 1) + (" [%d]" % subpayment[1] if subpayment else "") + bcolours.ENDC)
            if subpayment:
                self._amp_out[id][0][i].append(path)
                if not AMP_RESEND:
                    if self._amp_out[id][2]:
                        self._amp_out[id][2] = False
                        fail_count += 1
                else:
                    # Need to re-send here instead of returning False, not in experiments so ignore for now
                    pass
            else:
                fail_count += 1
            return False

        send_amnts = calc_path_fees(G, path, amnt)
        index_mapping = {path[i].get_id(): i for i in range(len(path))}
        index_mapping["src"] = self
        type = "pay_htlc" if len(path) > 2 else "pay"
        if subpayment: ttl = self._amp_out[subpayment[0]][1]

        p = Packet(path, index_mapping, send_amnts, type,
            (subpayment[0], subpayment[1], subpayment[2], ttl) if subpayment else False)
        p.set_timestamp(sim_time + self._encryption_delay + dest.get_decryption_delay())

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
        global fail_count

        if k == 1:
            self._init_payment(G, dest, amnt, node_comm, False, test_mode=test_mode, debug=debug)
        else:  # AMP
            # Of form, [subpayment index, failed routes]
            subp_statuses = [[] for i in range(k)]
            id = uuid4()

            # After timeout, attempt at AMP cancelled
            ttl = sim_time + AMP_TIMEOUT
            self._amp_out[id] = [subp_statuses, ttl, True]  # True here means hasn't failed yet

            routes = [p for p in self._find_path(G, dest, amnt / k, [], k=k)]

            if k > 1 and len(routes) != k:  # AMP, didn't find any route
                fail_count += 1
            else:
                # Send off each subpayment - first attempt.
                for i in range(k):
                    self._init_payment(G, dest, amnt / k, node_comm, (id, i, k), routes=routes, route_index=i, test_mode=test_mode, debug=debug)

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

    Messages are synced to simulator time and network latency is applied by using future timestamps.
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

        try:
            packet.set_timestamp(packet.get_timestamp() + self._latencies[(src, dest)])
        except KeyError:
            print("ERROR - packet sending to self... [%s/%s/%s] [Len: %d]" % (src, dest, packet.get_type(), len(packet.get_path())))
            return

        dest.receive_packet(packet, self._debug)


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
        fees = G[path[i]][path[i+1]]["fees"]
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

def _dijkstra_reverse(G, source, target, initial_amnt, avoid_edges=None):
    """Find a path from source to target, starting the search from the target towards source.

    Only supports directed graph.

    Constaint: path must be able to support amnt with compounded fees.

    Uses Dijkstra's algorithm and the following is adapted from NetworkX src code.

    Args:
        G: NetworkX graph in use.
        source: (Node) source node.
        target: (Node) destination node.
        initial_amnt: (float) amnt of satoshi intended to be transferred on the last hop to the target.
        avoid_edges: (list<Node>) when path finding, increase weight of these edges to avoid.

    Returns:
        the length of the shortest path found, using the given weight function and,
        the corresponding shortest path found.
    """
    source, target = target, source  # Search from target instead

    # Weight - Fee calculation on directed edge for a given amnt
    weight = lambda d, amnt: d["fees"][0] + d["fees"][1] * amnt

    Gsucc = G.successors

    push = heappush
    pop = heappop
    dist = {}  # Dictionary of final distances
    seen = {}
    paths = {source: [source]}

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
            cost = 0 if v == source else weight(G[w][v], seen[v] + initial_amnt)
            vw_dist = dist[v] + cost

            if avoid_edges and (w, v) in avoid_edges:
                vw_dist += 1000  # Bump up cost to avoid these edges

            if w not in seen or vw_dist < seen[w]:
                # Add or update if new shortest distance from src -> u
                # But only if the path "supports" the amnt with fees
                # If this is the last hop, we know the equity at one side of the channel
                if (w == target and cost + initial_amnt <= G[w][v]["equity"]) or \
                    (w != target and cost + initial_amnt <= G[w][v]["public_equity"] + G[v][w]["public_equity"]):
                    seen[w] = vw_dist
                    push(fringe, (vw_dist, next(c), w))
                    paths[w] = (paths[v] + [w])
    if target in dist:
        # Reverse the path!
        return dist[target], paths[target][::-1]
    else:
        # No path found
        return False

def _jit_dijsktra_reverse(G, source, jit_target, initial_amnt, full_knowledge=JIT_FULL_KNOWLEDGE):
    """Find a path from source to target, starting the search from the target towards source.

    This is for finding JIT routing.

    The actual target here is the source (cycle), but target should be the last other node.

    Constaint: path must be able to support amnt with compounded fees.

    Uses Dijkstra's algorithm and the following is adapted from NetworkX src code.

    Args:
        G: NetworkX graph in use.
        source: (Node) source node.
        jit_target: (Node) second last node before returning to source.
        initial_amnt: (float) amnt of satoshi intended to be transferred on the last hop to the target.

    Returns:
        the length of the shortest path found, using the given weight function and,
        the corresponding shortest path found.
    """
    # Source is "source" and "destination" - first hop must be jit_target (reversed).

    # Weight - Fee calculation on directed edge for a given amnt
    weight = lambda d, amnt: d["fees"][0] + d["fees"][1] * amnt

    Gsucc = G.successors

    push = heappush
    pop = heappop
    dist = {}  # Dictionary of final distances
    seen = {}
    paths = {source: [source]}

    # Fringe is heapq with 3-tuples (distance, c, node)
    # Use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []

    seen[source] = 0
    push(fringe, (0, next(c), source))

    first_src_popped = False

    while fringe:
        (d, _, v) = pop(fringe)
        if first_src_popped and v in dist:
            continue  # Already searched this node.
        dist[v] = d
        if first_src_popped and v == source:  # Found cycle
            break
        if v == source:
            first_src_pop = True
        for w in Gsucc(v):
            if v == source and w != jit_target:  # Needs to come from jit_target to source when reversed
                continue

            if v == jit_target and not G.has_edge(w, source):  # No cycle of len 3 exists
                continue

            if v != jit_target and v != source:  # Only go down the cycle path
                if w != source:
                    continue

            # No fees on last hop
            cost = 0 if v == source else weight(G[w][v], seen[v] + initial_amnt)
            vw_dist = dist[v] + cost

            if w not in seen or vw_dist < seen[w] or seen[w] == 0:
                # Add or update if new shortest distance from src -> u
                # But only if the path "supports" the amnt with fees
                # If this is the first or last hop, we know the equity at one side of the channel
                if ((w == jit_target or w == source or full_knowledge) and cost + initial_amnt <= G[w][v]["equity"]) or \
                    (not (w == jit_target or w == source or full_knowledge) and cost + initial_amnt <= G[w][v]["public_equity"] + G[v][w]["public_equity"]):
                    seen[w] = vw_dist
                    push(fringe, (vw_dist, next(c), w))
                    paths[w] = (paths[v] + [w])
    if source in dist:
        # Reverse the path!
        return dist[source], paths[source][::-1]
    else:
        # No path found
        return False

def _slack_based_reverse(G, source, target, initial_amnt):
    """Find a path from source to target, starting the search from the target towards source.

    Only supports directed graph.

    Constaint: path must be able to support amnt with compounded fees.

    Uses Dijkstra's algorithm and the following is adapted from NetworkX src code,
    but the weights are based on "slack" on channels, where path finding prefers
    routes whose total channel balances exceed the send amnt the most.

    Args:
        G: NetworkX graph in use.
        source: (Node) source node.
        target: (Node) destination node.
        initial_amnt: (float) amnt of satoshi intended to be transferred on the last hop to the target.

    Returns:
        the length of the shortest path found, using the given weight function and,
        the corresponding shortest path found.
    """
    source, target = target, source  # Search from target instead

    # Weight - Fee calculation on directed edge for a given amnt,
    # to know what paths we can travel down.
    fees_weight = lambda d, amnt: d["fees"][0] + d["fees"][1] * amnt

    Gsucc = G.successors

    push = heappush
    pop = heappop
    slacks = {}  # Dictionary of values relating to slack in path so far
    dist = {}  # Dictionary of distances so far
    seen = {}
    paths = {source: [source]}

    # Fringe is heapq with 6-tuples (slack val, c, node, distance, prev node in path, len path)
    # Use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []

    seen[source] = 0
    push(fringe, (0, next(c), source, 0, None, 0))

    while fringe:
        (slack_val, _, v, d, prev_node, len_path) = pop(fringe)
        slack_val = slack_val * -1
        if v in dist:
            continue  # Already searched this node.
        slacks[v] = slack_val
        dist[v] = d
        if v == target:
            continue
        for w in Gsucc(v):
            # No fees on last hop
            cost = 0 if v == source else fees_weight(G[w][v], seen[v] + initial_amnt)
            vw_dist = dist[v] + cost

            if v == target:
                new_slack_val = slack_val
            else:
                vw_slack = (G[w][v]["public_equity"] + G[v][w]["public_equity"]) - (vw_dist + initial_amnt)
                new_slack_val = (len_path * slack_val + vw_slack) / (len_path + 1)

            if len_path > (8 - 1):
                new_slack_val = -inf

            if w not in seen or new_slack_val > slacks[w]:
                # Add or update if new shortest distance from src -> u
                # But only if the path "supports" the amnt with fees
                # If this is the last hop, we know the equity at one side of the channel
                if (w == target and cost + initial_amnt <= G[w][v]["equity"]) or \
                    (w != target and cost + initial_amnt <= G[w][v]["public_equity"] + G[v][w]["public_equity"]):
                    seen[w] = vw_dist
                    slacks[w] = new_slack_val
                    # Negated so max heap instead of min heap - want highest slack
                    push(fringe, (-new_slack_val, next(c), w, vw_dist, v, len_path + 1))
                    paths[w] = (paths[v] + [w])
    if target in dist:
        # Reverse the path!
        return dist[target], paths[target][::-1]
    else:
        # No path found
        return False

def _bfs_reverse(G, source, target, initial_amnt):
    """Find a path from source to target, starting the search from the target towards source.

    Only supports directed graph.

    Constaint: path must be able to support amnt with compounded fees.

    Uses the breatdh first search algorithm.

    Args:
        G: NetworkX graph in use.
        source: (Node) source node.
        target: (Node) destination node.
        initial_amnt: (float) amnt of satoshi intended to be transferred on the last hop to the target.
    Returns:
        the length of the shortest hop path found, using the given weight function and,
        the corresponding shortest hop path found.
    """
    source, target = target, source  # Search from target instead

    # Weight - Fee calculation on directed edge for a given amnt
    weight = lambda d, amnt: d["fees"][0] + d["fees"][1] * amnt

    Gsucc = G.successors

    push = heappush
    pop = heappop
    dist = {}  # Dictionary of final distances
    seen = {}
    paths = {source: [source]}

    # Fringe is heapq with 3-tuples (const 1, c, node, distance)
    # Use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []

    seen[source] = 0
    push(fringe, (1, next(c), source, 0))

    while fringe:
        (_, _, v, d) = pop(fringe)
        if v in dist:
            continue  # Already searched this node.
        dist[v] = d
        if v == target:
            break
        for w in Gsucc(v):
            # No fees on last hop
            cost = 0 if v == source else weight(G[w][v], seen[v] + initial_amnt)
            vw_dist = dist[v] + cost

            if w not in seen:
                # Add or update if new shortest distance from src -> u
                # But only if the path "supports" the amnt with fees
                # If this is the last hop, we know the equity at one side of the channel
                if (w == target and cost + initial_amnt <= G[w][v]["equity"]) or \
                    (w != target and cost + initial_amnt <= G[w][v]["public_equity"] + G[v][w]["public_equity"]):
                    seen[w] = vw_dist
                    push(fringe, (1, next(c), w, vw_dist))
                    paths[w] = (paths[v] + [w])
    if target in dist:
        # Reverse the path!
        return dist[target], paths[target][::-1]
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
    encryption_delay = np.random.choice(ENCRYPTION_DELAYS, 1, p=ENCRYPTION_DELAY_DISTRIBUTION)[0]
    decryption_delay = np.random.choice(DECRYPTION_DELAYS, 1, p=DECRYPTION_DELAY_DISTRIBUTION)[0]

    return Node(i, merchant, spend_freq, receive_freq, encryption_delay, decryption_delay)

def generate_edge_args():
    """Generate random equity and fee distribution arguments for a new pair of edges. """
    amnt_a = np.random.choice([8000000, 500000], 1, p=[HIGH_FUNDS_CHANCE, 1 - HIGH_FUNDS_CHANCE])[0]
    amnt_b = np.random.choice([8000000, 500000], 1, p=[HIGH_FUNDS_CHANCE, 1 - HIGH_FUNDS_CHANCE])[0]

    # For search strat experiments
    # base_fee_a = np.random.choice([0, 0.0025, 1, 5], 1, p=[0.05, 0.2, 0.7, 0.05])[0]
    # base_fee_b = np.random.choice([0, 0.0025, 1, 5], 1, p=[0.05, 0.2, 0.7, 0.05])[0]
    #
    # fee_rate_a = np.random.choice([0, 0.00001, 0.00055, 0.001, 0.003], 1, p=[0.05, 0.45, 0.25, 0.2, 0.05])[0]
    # fee_rate_b = np.random.choice([0, 0.00001, 0.00055, 0.001, 0.003], 1, p=[0.05, 0.45, 0.25, 0.2, 0.05])[0]

    # Regular experiments
    base_fee_a = 1
    base_fee_b = 1

    fee_rate_a = base_fee_a * 0.000001
    fee_rate_b = base_fee_b * 0.000001

    dir_a = [amnt_a, [base_fee_a, fee_rate_a]]
    dir_b = [amnt_b, [base_fee_b, fee_rate_b]]

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
        nodes = [Node(i, False, 0, 0, 0, 0) for i in range(n)]
    else:
        nodes = [init_random_node(i) for i in range(n)]

    edge_pairs = set()

    # Connect each node to k/2 neighbours
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # First j nodes are now last in list
        pairs = list(zip(nodes, targets))
        for pair in pairs:
            if test:
                G.add_edge(pair[0], pair[1], equity=20, public_equity=20, fees=[0.1, 0.005], fee_metrics=[None, 0, 0])
                G.add_edge(pair[1], pair[0], equity=10, public_equity=10, fees=[0.1, 0.005], fee_metrics=[None, 0, 0])
            else:
                args = generate_edge_args()
                G.add_edge(pair[0], pair[1], equity=args[0][0], public_equity=args[0][0], fees=args[0][1], fee_metrics=[None, 0, 0])
                G.add_edge(pair[1], pair[0], equity=args[1][0], public_equity=args[1][0], fees=args[1][1], fee_metrics=[None, 0, 0])
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
                        G.add_edge(u, w, equity=20, public_equity=20, fees=[0.1, 0.005], fee_metrics=[None, 0, 0])
                        G.add_edge(w, u, equity=10, public_equity=10, fees=[0.1, 0.005], fee_metrics=[None, 0, 0])
                    else:
                        args = generate_edge_args()
                        G.add_edge(u, w, equity=args[0][0], public_equity=args[0][0], fees=args[0][1], fee_metrics=[None, 0, 0])
                        G.add_edge(w, u, equity=args[1][0], public_equity=args[1][0], fees=args[1][1], fee_metrics=[None, 0, 0])
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
            G.add_edge(pair[0], pair[1], equity=args[0][0], public_equity=args[0][0], fees=args[0][1], fee_metrics=[None, 0, 0])
            G.add_edge(pair[1], pair[0], equity=args[1][0], public_equity=args[1][0], fees=args[1][1], fee_metrics=[None, 0, 0])

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
    """Returns an appropriate spend amount (satoshi) for the selected node."""
    max_amnt = 800000

    return np.random.randint(1, max_amnt + 1)

def simulator(max_multiplier, send_timing, rb=0.0, hfc=0.2, k=1, ws=True):
    """Main simulator calling function. """
    global success_count
    global fail_count
    global sim_time
    global packet_queue
    global amp_retry_count
    global total_path_hops
    global total_paths_found
    global path_too_large_count

    global ROLE_BIAS
    ROLE_BIAS = rb

    global HIGH_FUNDS_CHANCE
    HIGH_FUNDS_CHANCE = hfc

    start_time = time.time()

    if ws:
        G, pairs = generate_wattz_strogatz(NUM_NODES, 10, 0.3)
        print("WS")
    else:
        G, pairs = generate_barabasi_albert(NUM_NODES, 5)
        print("BA")

    print("max_amnt:", max_multiplier * 100000)
    print("Timing:", send_timing)
    print("role bias:", ROLE_BIAS)
    print("High funds chance:", HIGH_FUNDS_CHANCE)
    print("k [AMP]:", k)
    print("JIT: [%s (repeat: %s // full knowledge: %s)]" % (JIT_ROUTING, JIT_REPEAT_REBALANCE, JIT_FULL_KNOWLEDGE))

    start_unbalance = calc_g_unbalance(G, pairs)

    nodes = list(G)
    node_comm = NodeCommunicator(nodes, pairs, debug=DEBUG)

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

    sent_payments = 0
    last_success = 0

    if DEBUG: print("// SIMULATION LOOP BEGIN\n-------------------------")

    in_flight_tracker = 0
    c = 0

    next_payment_time = 0

    show = True

    while success_count + fail_count < TOTAL_TO_SEND:
    # while success_count < 100 or success_count / (success_count + fail_count) >= 0.7:
        if (success_count + fail_count) % 5000 == 0:
            if show: print(success_count + fail_count, "sent")
            show = False
        else:
            show = True
        try:
            try:
                p, node = packet_queue.get(False)

                if p.get_timestamp() <= sim_time:
                    node.process_packet(G, node_comm, p)
                else:  # Put it back on
                    packet_queue.put((p, node))

                    if next_payment_time <= sim_time:
                        in_flight_tracker += (sent_payments - success_count - fail_count)
                        c += 1

                        selected = select_payment_nodes(consumers, merchants, total_freqs)

                        while selected[0] == selected[1]:
                            selected = select_payment_nodes(consumers, merchants, total_freqs)

                        max_amnt = max_multiplier * 100000
                        amnt = np.random.randint(1, max_amnt + 1)

                        selected[0].init_payment(G, selected[1], amnt, node_comm, k=k)

                        sent_payments += 1
                        next_payment_time += send_timing

                    sim_time = min(p.get_timestamp(), next_payment_time)
            except:  # Jump to next sending of a payment
                in_flight_tracker += (sent_payments - success_count - fail_count)
                c += 1

                selected = select_payment_nodes(consumers, merchants, total_freqs)

                while selected[0] == selected[1]:
                    selected = select_payment_nodes(consumers, merchants, total_freqs)

                max_amnt = max_multiplier * 100000
                amnt = np.random.randint(1, max_amnt + 1)

                selected[0].init_payment(G, selected[1], amnt, node_comm, k=k)

                sent_payments += 1
                next_payment_time += send_timing
        except KeyboardInterrupt:
            break


    print("// ANALYTICS\n-------------------------")
    print("# successful payments:", success_count)
    print("# failed payments:", fail_count)
    print("# amp retries:", amp_retry_count)
    print("Avg in flight:", in_flight_tracker / c)
    print("# Avg path hops:", total_path_hops / total_paths_found)
    print("Total paths over max len:", path_too_large_count)
    print("Time to run:", (time.time() - start_time) / 60)

    # Reset
    success_count = 0
    fail_count = 0
    sim_time = 0
    amp_retry_count = 0
    total_path_hops = 0
    total_paths_found = 0
    path_too_large_count = 0
    packet_queue = PriorityQueue()

"""
//////////////////////////
CORE FUNCTIONALITY TESTING
//////////////////////////
"""

def process_next_test_packet(G, node_comm, debug, type=False, amp_resend_enabled=False, jit_enabled=False, node_check=False):
    """Process the next packet when testing."""
    global packet_queue
    global sim_time

    p, node = packet_queue.get(False)
    if type: assert p.get_type() == type
    if node_check: assert node == node_check

    sim_time = p.get_timestamp()
    node.process_packet(G, node_comm, p, amp_resend_enabled=amp_resend_enabled, jit_enabled=jit_enabled, debug=debug)

def test_func():
    """Testing."""
    global sim_time

    sim_time = 0
    TEST_DEBUG = False

    G, pairs = generate_wattz_strogatz(NUM_TEST_NODES, 6, 0.3, 10, True)
    nodes = list(G)

    node_comm = NodeCommunicator(nodes, pairs, test_mode=True, debug=TEST_DEBUG)

    ####################################
    # PATHFINDING & FEES
    ####################################

    nodes_ = []
    for i in range(8):
        nodes_.append(init_random_node(i))

    G_ = nx.DiGraph()
    G_.add_nodes_from(nodes_)

    G_.add_edge(nodes_[0], nodes_[1], equity=150, public_equity=150, fees=[1, 0.001])
    G_.add_edge(nodes_[1], nodes_[0], equity=50, public_equity=50, fees=[1, 0.001])

    G_.add_edge(nodes_[1], nodes_[2], equity=100, public_equity=100, fees=[2, 0.001])
    G_.add_edge(nodes_[2], nodes_[1], equity=50, public_equity=50, fees=[1, 0.001])

    G_.add_edge(nodes_[1], nodes_[3], equity=20, public_equity=20, fees=[2, 0.001])
    G_.add_edge(nodes_[3], nodes_[1], equity=80, public_equity=80, fees=[1, 0.001])

    G_.add_edge(nodes_[2], nodes_[6], equity=100, public_equity=100, fees=[3, 0.001])
    G_.add_edge(nodes_[6], nodes_[2], equity=50, public_equity=50, fees=[1, 0.001])

    G_.add_edge(nodes_[3], nodes_[4], equity=50, public_equity=50, fees=[5, 0.001])
    G_.add_edge(nodes_[4], nodes_[3], equity=100, public_equity=100, fees=[1, 0.001])

    G_.add_edge(nodes_[3], nodes_[5], equity=30, public_equity=30, fees=[2, 0.001])
    G_.add_edge(nodes_[5], nodes_[3], equity=160, public_equity=160, fees=[1, 0.001])

    G_.add_edge(nodes_[4], nodes_[5], equity=20, public_equity=20, fees=[1, 0.001])
    G_.add_edge(nodes_[5], nodes_[4], equity=90, public_equity=90, fees=[1, 0.001])

    G_.add_edge(nodes_[4], nodes_[6], equity=40, public_equity=40, fees=[1, 0.001])
    G_.add_edge(nodes_[6], nodes_[4], equity=110, public_equity=110, fees=[2, 0.001])

    G_.add_edge(nodes_[5], nodes_[7], equity=100, public_equity=100, fees=[1, 0.001])
    G_.add_edge(nodes_[7], nodes_[5], equity=50, public_equity=50, fees=[1, 0.001])

    G_.add_edge(nodes_[0], nodes_[7], equity=40, public_equity=40, fees=[1, 0.001])
    G_.add_edge(nodes_[7], nodes_[0], equity=40, public_equity=40, fees=[1, 0.001])


    ################## func: _dijkstra_reverse & _find_path
    cost, path = _dijkstra_reverse(G_, nodes_[0], nodes_[7], 90)
    assert floor_msat(cost) == 5.2762
    assert path == [nodes_[0], nodes_[1], nodes_[3], nodes_[5], nodes_[7]]

    # New path found, as independent as possible
    path = [p for p in nodes_[0]._find_path(G_, nodes_[7], 90, [path])][0]
    assert path == [nodes_[0], nodes_[1], nodes_[2], nodes_[6], nodes_[4], nodes_[5], nodes_[7]]

    cost, path = _dijkstra_reverse(G_, nodes_[0], nodes_[7], 100)
    assert floor_msat(cost) == 9.519
    assert path == [nodes_[0], nodes_[1], nodes_[2], nodes_[6], nodes_[4], nodes_[5], nodes_[7]]

    cost, path = _dijkstra_reverse(G_, nodes_[0], nodes_[7], 110)
    assert floor_msat(cost) == 11.6896
    assert path == [nodes_[0], nodes_[1], nodes_[2], nodes_[6], nodes_[4], nodes_[3], nodes_[5], nodes_[7]]

    # Funds distribution on first hop known (last hop pathfinding)
    attempt = _dijkstra_reverse(G_, nodes[0], nodes_[7], 160)
    assert attempt == False


    ################## func: _bfs_reverse
    _, path = _bfs_reverse(G_, nodes_[0], nodes_[7], 10)
    assert path == [nodes_[0], nodes_[7]]

    _, path = _bfs_reverse(G_, nodes_[0], nodes_[7], 90)
    assert path == [nodes_[0], nodes_[1], nodes_[3], nodes_[5], nodes_[7]]


    ################## func: _slack_based_reverse
    _, path = _slack_based_reverse(G_, nodes_[0], nodes_[7], 10)
    assert path == [nodes_[0], nodes_[1], nodes_[2], nodes_[6], nodes_[4], nodes_[3], nodes_[5], nodes_[7]]

    # Boost to change path
    G_[nodes_[4]][nodes_[5]]["public_equity"] = 250

    _, path = _slack_based_reverse(G_, nodes_[0], nodes_[7], 10)
    assert path == [nodes_[0], nodes_[1], nodes_[2], nodes_[6], nodes_[4], nodes_[5], nodes_[7]]


    ################## func: _jit_dijsktra_reverse
    nodes_.append(init_random_node(8))

    G_.add_edge(nodes_[3], nodes_[8], equity=40, public_equity=40, fees=[1, 0.001])
    G_.add_edge(nodes_[8], nodes_[3], equity=40, public_equity=40, fees=[1, 0.001])

    G_.add_edge(nodes_[5], nodes_[8], equity=40, public_equity=40, fees=[1, 0.001])
    G_.add_edge(nodes_[8], nodes_[5], equity=0, public_equity=40, fees=[10, 0.001])

    _, path = _jit_dijsktra_reverse(G_, nodes_[3], nodes_[5], 10)
    assert len(path) == 4  # 3 hops
    assert path == [nodes_[3], nodes_[4], nodes_[5], nodes_[3]]

    # Reduce fees
    G_[nodes_[8]][nodes_[5]]["fees"] = [0.5, 0.001]

    _, path = _jit_dijsktra_reverse(G_, nodes_[3], nodes_[5], 10)
    assert path == [nodes_[3], nodes_[8], nodes_[5], nodes_[3]]

    # Reverts when full knowledge, as equity from 0 -> 5 is zero
    _, path = _jit_dijsktra_reverse(G_, nodes_[3], nodes_[5], 10, full_knowledge=True)
    assert path == [nodes_[3], nodes_[4], nodes_[5], nodes_[3]]


    ################## func: calc_path_fees
    # Fees are base 0.1 and 0.005 multiplier on test set, arbitarily chosen.
    path = [nodes[0], nodes[3], nodes[18], nodes[19]]  # Existing path, 3 hops
    hop_amnts = calc_path_fees(G, path, 10)  # Amount sent per hop

    assert hop_amnts[0] == 10.3007  # 10 + 0.1 + 0.005 * 10.15 (amount sent on hop ahead) - floored to msat
    assert hop_amnts[1] == 10.15  # 10 + 0.1 + 0.005 * 10 (original payment amount)
    assert hop_amnts[2] == 10  # No fees on last hop



    ####################################
    # REGULAR PAYMENTS
    ####################################

    ################## single-hop: success & failure
    e_0_19 = G[nodes[0]][nodes[19]]["equity"]
    nodes[0].init_payment(G, nodes[19], 10, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[0].init_payment(G, nodes[19], 10, node_comm, test_mode=True, debug=TEST_DEBUG)

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - 10  # Success

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    assert G[nodes[0]][nodes[19]]["equity"] == e_0_19 - 10  # Hasn't changed - failure


    ################## multi-hop: success [Path: 0 - 2 - 4]
    e_0_2 = G[nodes[0]][nodes[2]]["equity"]
    e_2_0 = G[nodes[2]][nodes[0]]["equity"]
    e_2_4 = G[nodes[2]][nodes[4]]["equity"]
    e_4_2 = G[nodes[4]][nodes[2]]["equity"]
    nodes[0].init_payment(G, nodes[4], 10, node_comm, test_mode=True, debug=TEST_DEBUG)

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    assert G[nodes[0]][nodes[2]]["equity"] == e_0_2 - 10.15  # Reduced
    assert G[nodes[2]][nodes[0]]["equity"] == e_2_0  # Hasn't increased yet

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    assert G[nodes[2]][nodes[4]]["equity"] == e_2_4 - 10  # Reduced
    assert G[nodes[4]][nodes[2]]["equity"] == e_4_2 + 10  # Automatically claimed

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    assert G[nodes[2]][nodes[0]]["equity"] == e_2_0 + 10.15  # Claimed

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)  # Complete proof of payment


    ################## multi-hop: fail by insufficient funds @ start [Path: 0 - 2 - 4]
    G[nodes[2]][nodes[4]]["equity"] = 2  # Reduce so path will fail

    e_0_2 = G[nodes[0]][nodes[2]]["equity"]
    e_2_0 = G[nodes[2]][nodes[0]]["equity"]
    e_2_4 = G[nodes[2]][nodes[4]]["equity"]
    e_4_2 = G[nodes[4]][nodes[2]]["equity"]
    nodes[0].init_payment(G, nodes[4], 5, node_comm, test_mode=True, debug=TEST_DEBUG)

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    assert G[nodes[0]][nodes[2]]["equity"] == e_0_2 - 5.125  # Reduced
    assert G[nodes[2]][nodes[0]]["equity"] == e_2_0  # Hasn't increased yet

    # Cancel immediately, not cancel_rest
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, type="cancel")

    # Claimed back, and all return original amounts
    assert G[nodes[0]][nodes[2]]["equity"] == e_0_2
    assert G[nodes[2]][nodes[0]]["equity"] == e_2_0
    assert G[nodes[2]][nodes[4]]["equity"] == e_2_4
    assert G[nodes[4]][nodes[2]]["equity"] == e_4_2


    ################## multi-hop: fail by funds reduced mid-payment [Path: 0 - 2 - 4]
    G[nodes[2]][nodes[4]]["equity"] = 10  # Revert

    e_0_2 = G[nodes[0]][nodes[2]]["equity"]
    e_2_0 = G[nodes[2]][nodes[0]]["equity"]
    e_2_4 = G[nodes[2]][nodes[4]]["equity"]
    e_4_2 = G[nodes[4]][nodes[2]]["equity"]
    nodes[0].init_payment(G, nodes[4], 5, node_comm, test_mode=True, debug=TEST_DEBUG)

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    assert G[nodes[0]][nodes[2]]["equity"] == e_0_2 - 5.125  # Reduced
    assert G[nodes[2]][nodes[0]]["equity"] == e_2_0  # Hasn't increased yet

    G[nodes[2]][nodes[4]]["equity"] = 2  # Reduce so path will fail - mid-payment
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)

    # Propogate cancel back, no HTLC on first cancellation, so cancel_rest first
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, type="cancel_rest")
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, type="cancel")

    # Claimed back, and all return original amounts
    assert G[nodes[0]][nodes[2]]["equity"] == e_0_2
    assert G[nodes[2]][nodes[0]]["equity"] == e_2_0
    assert G[nodes[2]][nodes[4]]["equity"] == 2
    assert G[nodes[4]][nodes[2]]["equity"] == e_4_2


    ################## packet ordering [Paths: 8 - 6 - 16, 9 - 11]
    node_comm.set_latency(nodes[8], nodes[6], 10)
    nodes[8].init_payment(G, nodes[16], 5, node_comm, test_mode=True, debug=TEST_DEBUG)
    nodes[9].init_payment(G, nodes[11], 5, node_comm, test_mode=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, node_check=nodes[11], debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)



    ####################################
    # JIT ROUTING
    ####################################

    ################## JIT Routing - payment succeeds that would otherwise fail [Path: 0 - 1 - 2]
    nodes[0].init_payment(G, nodes[2], 15, node_comm, test_mode=True, debug=TEST_DEBUG)
    G[nodes[1]][nodes[2]]["equity"] = 10  # Will fail, needs to send 15

    e_0_1 = G[nodes[0]][nodes[1]]["equity"]
    e_1_0 = G[nodes[1]][nodes[0]]["equity"]
    e_1_2 = G[nodes[1]][nodes[2]]["equity"]
    e_2_1 = G[nodes[2]][nodes[1]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - 15.175
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0

    # Re-balance [Path: 1 - 3 - 2 - 1]
    e_1_3 = G[nodes[1]][nodes[3]]["equity"]
    e_3_1 = G[nodes[3]][nodes[1]]["equity"]
    e_3_2 = G[nodes[3]][nodes[2]]["equity"]
    e_2_3 = G[nodes[2]][nodes[3]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    assert G[nodes[1]][nodes[2]]["equity"] == e_1_2 + 5  # More available now

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    assert G[nodes[1]][nodes[3]]["equity"] == e_1_3 - 5.2506
    assert G[nodes[3]][nodes[1]]["equity"] == e_3_1 + 5.2506
    assert G[nodes[3]][nodes[2]]["equity"] == e_3_2 - 5.125
    assert G[nodes[2]][nodes[3]]["equity"] == e_2_3 + 5.125

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    assert G[nodes[0]][nodes[1]]["equity"] == e_0_1 - 15.175
    assert G[nodes[1]][nodes[0]]["equity"] == e_1_0 + 15.175
    assert G[nodes[1]][nodes[2]]["equity"] == e_1_2 - 10
    assert G[nodes[2]][nodes[1]]["equity"] == e_2_1 + 10


    ################## JIT Routing - fails - no re-balance route found [Path: 0 - 3 - 18 - 19]
    nodes[0].init_payment(G, nodes[19], 15, node_comm, test_mode=True, debug=TEST_DEBUG)
    G[nodes[3]][nodes[15]]["equity"] = 0
    G[nodes[3]][nodes[1]]["equity"] = 0
    G[nodes[3]][nodes[5]]["equity"] = 0

    e_0_3 = G[nodes[0]][nodes[3]]["equity"]
    e_3_0 = G[nodes[3]][nodes[0]]["equity"]
    e_3_18 = G[nodes[3]][nodes[18]]["equity"]
    e_18_3 = G[nodes[18]][nodes[3]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    # All reverted
    assert G[nodes[0]][nodes[3]]["equity"] == e_0_3
    assert G[nodes[3]][nodes[0]]["equity"] == e_3_0
    assert G[nodes[3]][nodes[18]]["equity"] == e_3_18
    assert G[nodes[18]][nodes[3]]["equity"] == e_18_3


    ################## JIT Routing - failed rebalance [Path: 0 - 3 - 18 - 19]
    nodes[0].init_payment(G, nodes[19], 15, node_comm, test_mode=True, debug=TEST_DEBUG)
    G[nodes[3]][nodes[15]]["equity"] = 20  # Revert so works
    G[nodes[15]][nodes[18]]["equity"] = 0  # So re-balance fails

    e_0_3 = G[nodes[0]][nodes[3]]["equity"]
    e_3_0 = G[nodes[3]][nodes[0]]["equity"]
    e_3_18 = G[nodes[3]][nodes[18]]["equity"]
    e_18_3 = G[nodes[18]][nodes[3]]["equity"]
    e_3_15 = G[nodes[3]][nodes[15]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    assert G[nodes[3]][nodes[15]]["equity"] - (e_3_15 - 5.4273) <= 0.0001  # HTLC set up (rouding errors)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    # All reverted
    assert G[nodes[3]][nodes[15]]["equity"] == e_3_15
    assert G[nodes[0]][nodes[3]]["equity"] == e_0_3
    assert G[nodes[3]][nodes[0]]["equity"] == e_3_0
    assert G[nodes[3]][nodes[18]]["equity"] == e_3_18
    assert G[nodes[18]][nodes[3]]["equity"] == e_18_3


    ################## JIT Routing - funds depleted futher - failed re-rebalance [Path: 0 - 3 - 18 - 19]
    nodes[0].init_payment(G, nodes[19], 15, node_comm, test_mode=True, debug=TEST_DEBUG)
    G[nodes[3]][nodes[15]]["equity"] = 20  # Revert so works
    G[nodes[15]][nodes[18]]["equity"] = 20

    e_0_3 = G[nodes[0]][nodes[3]]["equity"]
    e_3_0 = G[nodes[3]][nodes[0]]["equity"]
    e_3_18 = G[nodes[3]][nodes[18]]["equity"]
    e_18_3 = G[nodes[18]][nodes[3]]["equity"]
    e_3_15 = G[nodes[3]][nodes[15]]["equity"]
    e_15_3 = G[nodes[15]][nodes[3]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    assert G[nodes[3]][nodes[15]]["equity"] - (e_3_15 - 5.4273) <= 0.0001  # HTLC set up (rouding errors)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    G[nodes[3]][nodes[18]]["equity"] -= 5  # Deplete further
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    G[nodes[15]][nodes[18]]["equity"] = 5  # Reduce to second re-balance won't work

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    # All reverted, but one re-balance worked
    assert G[nodes[3]][nodes[15]]["equity"] - (e_3_15 - 5.4273) <= 0.0001
    assert G[nodes[15]][nodes[3]]["equity"] - (e_15_3 + 5.4273) <= 0.0001
    assert G[nodes[0]][nodes[3]]["equity"] == e_0_3
    assert G[nodes[3]][nodes[0]]["equity"] == e_3_0
    assert G[nodes[3]][nodes[18]]["equity"] == e_3_18 - 5 + 5.175  # minus 5 for manual depletion
    assert G[nodes[18]][nodes[3]]["equity"] == e_18_3 - 5.175


    ################## JIT Routing - funds depleted futher - successful re-rebalance [Path: 0 - 3 - 18 - 19]
    nodes[0].init_payment(G, nodes[19], 15, node_comm, test_mode=True, debug=TEST_DEBUG)
    G[nodes[3]][nodes[15]]["equity"] = 20  # Revert so works
    G[nodes[15]][nodes[18]]["equity"] = 20

    e_0_3 = G[nodes[0]][nodes[3]]["equity"]
    e_3_0 = G[nodes[3]][nodes[0]]["equity"]
    e_3_18 = G[nodes[3]][nodes[18]]["equity"]
    e_18_3 = G[nodes[18]][nodes[3]]["equity"]
    e_18_19 = G[nodes[18]][nodes[19]]["equity"]
    e_19_18 = G[nodes[19]][nodes[18]]["equity"]

    e_3_15 = G[nodes[3]][nodes[15]]["equity"]
    e_15_3 = G[nodes[15]][nodes[3]]["equity"]
    e_15_18 = G[nodes[15]][nodes[18]]["equity"]
    e_18_15 = G[nodes[18]][nodes[15]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    assert G[nodes[3]][nodes[15]]["equity"] == e_3_15 - 5.2506  # HTLC set up
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    G[nodes[3]][nodes[18]]["equity"] -= 5  # Deplete further
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG, jit_enabled=True)

    # All reverted, but one re-balance worked
    assert G[nodes[0]][nodes[3]]["equity"] - (e_0_3 - 15.3508) <= 0.0001
    assert G[nodes[3]][nodes[0]]["equity"] - (e_0_3 + 15.3508) <= 0.0001
    assert G[nodes[3]][nodes[18]]["equity"] == e_3_18 - 5 - 15.175 + 5 + 5  # minus 5 for manual depletion
    assert G[nodes[18]][nodes[3]]["equity"] == e_18_3 + 15.175 - 5 - 5
    assert G[nodes[18]][nodes[19]]["equity"] == e_18_19 - 15
    assert G[nodes[19]][nodes[18]]["equity"] == e_19_18 + 15

    assert G[nodes[3]][nodes[15]]["equity"] == e_3_15 - 5.2506 - 5.2506
    assert G[nodes[15]][nodes[3]]["equity"] == e_15_3 + 5.2506 + 5.2506
    assert G[nodes[15]][nodes[18]]["equity"] == e_15_18 - 5.1250 - 5.1250
    assert G[nodes[18]][nodes[15]]["equity"] == e_18_15 + 5.1250 + 5.1250



    ####################################
    # ATOMIC MULTI-PATH PAYMENTS
    ####################################

    ################## successful multi-part payment, first go [Paths 7 - 9 - 12, 7 - 10 - 14 - 12]
    nodes[7].init_payment(G, nodes[12], 10, node_comm, k=2, test_mode=True, debug=TEST_DEBUG)

    e_9_12 = G[nodes[9]][nodes[12]]["equity"]
    e_12_9 = G[nodes[12]][nodes[9]]["equity"]
    e_12_14 = G[nodes[12]][nodes[14]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)

    # First received, final funds shouldn't be auto-claimed
    assert G[nodes[9]][nodes[12]]["equity"] == e_9_12 - 5
    assert G[nodes[12]][nodes[9]]["equity"] == e_12_9

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)

    # Both received, funds claimed
    assert G[nodes[12]][nodes[9]]["equity"] == e_12_9 + 5
    assert G[nodes[12]][nodes[14]]["equity"] == e_12_14 + 5

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)


    ################## failed multi-part payment, first go [Paths 7 - 9 - 12, 7 - 10 - 14 - 12]
    nodes[7].init_payment(G, nodes[12], 10, node_comm, k=2, test_mode=True, debug=TEST_DEBUG)
    G[nodes[10]][nodes[14]]["equity"] = 0  # Deplete so fails

    e_9_12 = G[nodes[9]][nodes[12]]["equity"]
    e_12_9 = G[nodes[12]][nodes[9]]["equity"]
    e_12_14 = G[nodes[12]][nodes[14]]["equity"]

    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, debug=TEST_DEBUG)

    # Fail, funds never claimed
    assert G[nodes[12]][nodes[9]]["equity"] == e_12_9
    assert G[nodes[12]][nodes[14]]["equity"] == e_12_14


    ################## successful multi-part payment with resending [Paths 7 - 9 - 12, 7 - 10 - 14 - 12 [-> 7 - 8 - 11 - 13 - 12]]
    nodes[7].init_payment(G, nodes[12], 10, node_comm, k=2, test_mode=True, debug=TEST_DEBUG)

    e_9_12 = G[nodes[9]][nodes[12]]["equity"]
    e_12_9 = G[nodes[12]][nodes[9]]["equity"]
    e_12_13 = G[nodes[12]][nodes[13]]["equity"]

    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)
    process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)

    # succeeds, funds never claimed
    assert G[nodes[12]][nodes[9]]["equity"] == e_12_9 + 5
    assert G[nodes[12]][nodes[13]]["equity"] == e_12_13 + 5


    ################## failed multi-part payment with resending (no routes left) [Paths 7 - 10 - 14 - 12 [-> all fail], 7 - 8 - 11 - 13 - 12]
    nodes[7].init_payment(G, nodes[12], 10, node_comm, k=2, test_mode=True, debug=TEST_DEBUG)

    e_9_12 = G[nodes[9]][nodes[12]]["equity"]
    e_12_9 = G[nodes[12]][nodes[9]]["equity"]
    e_12_13 = G[nodes[12]][nodes[13]]["equity"]

    # To ensure all paths fail
    G[nodes[10]][nodes[9]]["equity"] = 0
    G[nodes[10]][nodes[1]]["equity"] = 0
    G[nodes[10]][nodes[8]]["equity"] = 0

    for _ in range(22):
        process_next_test_packet(G, node_comm, amp_resend_enabled=True, debug=TEST_DEBUG)

    # succeeds, funds never claimed
    assert G[nodes[12]][nodes[9]]["equity"] == e_12_9
    assert G[nodes[12]][nodes[13]]["equity"] == e_12_13

if __name__ == "__main__":
    test_func()


for i in range(20):
    for max_amnt in [2]:
        for timing in [0.02]:
            for k in [1]:
                simulator(max_amnt, timing, k=k, ws=True)
                print("\n\n")
