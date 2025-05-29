def bitcount(n):
    count = 0
    while n > 0:  # Corrected condition
        n &= n - 1
        count += 1
    return count