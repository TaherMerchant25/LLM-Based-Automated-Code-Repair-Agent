def reverse_linked_list(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node = nextnode
        node.successor = prevnode
        prevnode = node
    return prevnode