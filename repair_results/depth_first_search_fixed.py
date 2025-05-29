def depth_first_search(startnode, goalnode):
    nodesvisited = set()
 
    def search_from(node):
        if node is goalnode:
            return True
        else:
            result = any(
                search_from(nextnode) for nextnode in node.successors
            )
            nodesvisited.add(node)
            return result

    return search_from(startnode)