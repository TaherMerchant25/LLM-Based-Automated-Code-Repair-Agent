def minimum_spanning_tree(weight_by_edge):
    group_by_node = {}
    mst_edges = set()

    for edge in sorted(weight_by_edge, key=weight_by_edge.__getitem__):
        u, v = edge
        if group_by_node.get(u, {u}) != group_by_node.get(v, {v}):
            mst_edges.add(edge)
            group_by_node[u] = group_by_node.get(u, {u}).union(group_by_node.get(v, {v}))
            group_by_node[v] = group_by_node[u]

    return mst_edges