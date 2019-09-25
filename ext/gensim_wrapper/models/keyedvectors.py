from gensim.models.keyedvectors import *


class KeyedVectorsWrapper(KeyedVectors):

    def __init__(self):
        super(KeyedVectorsWrapper, self).__init__()
        self.syn0_community = []
        self.syn0norm = None

    def save_word2vec_community_format(self, fname, fvocab=None, binary=False, total_vec=None):

        number_of_communities = self.syn0_community.shape[0]

        if total_vec is None:
            total_vec = len(self.vocab)
        vector_size = self.syn0.shape[1]

        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (number_of_communities, vector_size)))
            # store in sorted order: most frequent words at the top
            for t in range(number_of_communities):
                row = self.syn0_community[t]

                fout.write(utils.to_utf8("%s %s\n" % (t, ' '.join("%f" % val for val in row))))