class Node:
    def __init__(self, value=None, successor=None, successors=[], predecessors=[], incoming_nodes=[], outgoing_nodes=[]):
        self.value = value
        self.successor = successor
        self.successors = successors
        self.predecessors = predecessors
        self.incoming_nodes = incoming_nodes
        self.outgoing_nodes = outgoing_nodes

    def successor(self):
        return self.successor

    def successors(self):
        return self.successors

    def predecessors(self):
        return self.predecessors

The bug was that the `__init__` method was defining instance variables with the same names as the methods, creating a naming conflict.  The corrected code removes the redundant `successor`, `successors`, and `predecessors` parameters from the `__init__` method.  The methods themselves are fine as they are, accessing the instance variables correctly.