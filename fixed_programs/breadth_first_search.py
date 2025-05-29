from collections import deque as Queue
 
def breadth_first_search(startnode, goalnode):
    queue = Queue()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while queue: #Corrected line
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            queue.extend(neighbor for neighbor in node.successors if neighbor not in nodesseen)
            nodesseen.update(neighbor for neighbor in node.successors if neighbor not in nodesseen)

    return False