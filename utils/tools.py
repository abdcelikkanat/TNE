

def read_corpus_file(corpus_path):
    walks = []
    with open(corpus_path, 'r') as f:
        for line in f.readlines():
            walk = [str(w) for w in line.strip().split()]
            walks.append(walk)

    return walks



