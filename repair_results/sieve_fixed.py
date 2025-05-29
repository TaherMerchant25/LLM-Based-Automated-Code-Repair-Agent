def sieve(max):
    primes = []
    is_prime = [True] * (max + 1)
    for p in range(2, int(max**0.5) + 1):
        if is_prime[p]:
            for i in range(p * p, max + 1, p):
                is_prime[i] = False
    for p in range(2, max + 1):
        if is_prime[p]:
            primes.append(p)
    return primes