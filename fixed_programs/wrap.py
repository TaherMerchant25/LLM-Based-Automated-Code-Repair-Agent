def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = text.rfind(' ', 0, cols + 1)
        if end == -1:
            end = cols
        line, text = text[:end], text[end+1:] #fixed this line
        lines.append(line)
    lines.append(text) #added this line
    return lines