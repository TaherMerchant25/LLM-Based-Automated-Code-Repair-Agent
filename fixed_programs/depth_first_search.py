def depth_first_search(startnode, goalnode):
    nodesvisited = set()
 
    def search_from(node):
        if node is goalnode:
            return True
        nodesvisited.add(node) #Added here to ensure all paths from a node are explored before marking it as visited.
        result = False
        for nextnode in node.successors:
            if nextnode not in nodesvisited: #Check for visited nodes before recursive call
                result = result or search_from(nextnode)
        return result

    return search_from(startnode)