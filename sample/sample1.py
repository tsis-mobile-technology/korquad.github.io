"""
02) 워드투벡터(Word2Vec)
앞서 원-핫 벡터는 단어 벡터 간 유의미한 유사도를 계산할 수 없다는 단점이 있음을 언급한 적이 있습니다.
그래서 단어 벡터 간 유의미한 유사도를 반영할 수 있도록 단어의 의미를 수치화 할 수 있는 방법이 필요합니다.
이를 위해서 사용되는 대표적인 방법이 워드투벡터(Word2Vec)입니다.

분산 표현(distributed representation) 방법은 기본적으로 분포 가설(distributional hypothesis)이라는 가정 하에 만들어진 표현 방법입니다.
이 가정은 '비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다' 라는 가정입니다.
강아지란 단어는 귀엽다, 예쁘다, 애교 등의 단어가 주로 함께 등장하는데 분포 가설에 따라서 해당 내용을 가진 텍스트의 단어들을 벡터화한다면 해당 단어 벡터들은 유사한 벡터값을 가집니다.
분산 표현은 분포 가설을 이용하여 텍스트를 학습하고, 단어의 의미를 벡터의 여러 차원에 분산하여 표현합니다.
"""
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