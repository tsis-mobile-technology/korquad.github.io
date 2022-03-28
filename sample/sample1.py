from gensim.models.word2vec import Word2Vec
import gensim

path = 'text8'
sentences = gensim.models.word2vec.Text8Corpus(path)

# model = Word2Vec(sentences, min_count=5, size=100, window=5)
model = Word2Vec(sentences, min_count=5, vector_size=100, window=5)
model.save('w2v_model')

saved_model = Word2Vec.load('w2v_model')

word_vector = saved_model['philosophy']

saved_model.similarity('philosophy', 'political')

saved_model.similar_by_word('philosophy')