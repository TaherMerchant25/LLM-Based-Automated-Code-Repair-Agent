def reverse_linked_list(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        prevnode = node #This line was missing, causing the prevnode to not update correctly.
        node = nextnode
    return prevnode