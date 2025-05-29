def next_palindrome(digit_list):
    n = len(digit_list)
    high_mid = n // 2
    low_mid = (n - 1) // 2
    carry = 1
    while high_mid >= 0 and low_mid < n and carry:
        sum_digits = digit_list[high_mid] + carry
        digit_list[high_mid] = sum_digits % 10
        carry = sum_digits // 10
        if high_mid != low_mid:
            digit_list[low_mid] = digit_list[high_mid]
        high_mid -= 1
        low_mid += 1

    if carry:
        result = [1] + [0] * n + [1]
        if n%2 != 0:
            result[n//2+1] = 0
        return result
    else:
        return digit_list