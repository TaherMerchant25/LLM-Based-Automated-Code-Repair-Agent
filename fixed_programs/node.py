class Node:
    def __init__(self, value=None, successors=[], predecessors=[]):
        self.value = value
        self.successors = successors
        self.predecessors = predecessors

    def get_successors(self):
        return self.successors

    def get_predecessors(self):
        return self.predecessors