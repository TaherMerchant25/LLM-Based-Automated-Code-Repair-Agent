def levenshtein(source, target):
    if source == '' or target == '':
        return len(source) or len(target)

    elif source[0] == target[0]:
        return levenshtein(source[1:], target[1:])

    else:
        return 1 + min(
            levenshtein(source,     target[1:]),
            levenshtein(source[1:], target[1:]),
            levenshtein(source[1:], target)
        )

The original code had a bug in the base case.  It incorrectly returned `1 + levenshtein(source[1:], target[1:])` when the first characters matched.  This should simply return the Levenshtein distance of the remaining substrings, without adding 1.  The corrected code removes the `1 +` from this line.