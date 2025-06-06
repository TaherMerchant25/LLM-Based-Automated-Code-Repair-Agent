def rpn_eval(tokens):
    def op(symbol, a, b):
        return {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b
        }[symbol](a, b)

    stack = []
 
    for token in tokens:
        if isinstance(token, float):
            stack.append(token)
        else:
            if len(stack) < 2:
                return 0 # Handle insufficient operands
            a = stack.pop()
            b = stack.pop()
            stack.append(
                op(token, a, b)
            )

    return stack.pop()