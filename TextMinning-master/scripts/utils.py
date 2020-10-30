def read_file_lines(addrs):
    results = []
    with open(addrs, 'r', encoding="utf_8") as f:
        stopwords = f.read()
        stopwords = stopwords.splitlines()
        for word in stopwords:
            results.append(word)

    return results
