def knapsack(capacity, items):
    from collections import defaultdict
    memo = defaultdict(int)

    for i in range(1, len(items) + 1):
        weight, value = items[i - 1]

        for j in range(capacity + 1): #Corrected line: range starts from 0
            memo[i, j] = memo[i - 1, j]

            if weight <= j: #Corrected line: should be less than or equal to
                memo[i, j] = max(
                    memo[i, j],
                    value + memo[i - 1, j - weight]
                )

    return memo[len(items), capacity]