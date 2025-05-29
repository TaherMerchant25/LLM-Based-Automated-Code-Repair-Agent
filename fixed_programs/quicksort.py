def quicksort(arr):
    if len(arr) < 2:
        return arr

    pivot = arr[len(arr) // 2]  # Choose pivot to mitigate worst-case scenarios
    lesser = []
    equal = []
    greater = []
    for x in arr:
        if x < pivot:
            lesser.append(x)
        elif x == pivot:
            equal.append(x)
        else:
            greater.append(x)

    return quicksort(lesser) + equal + quicksort(greater)

"""
QuickSort


Input:
    arr: A list of ints

Output:
    The elements of arr in sorted order
"""