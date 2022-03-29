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
# 3. 사전 훈련된 Word2Vec 임베딩(Pre-trained Word2Vec embedding) 소개
# 자연어 처리 작업을 할때, 케라스의 Embedding()를 사용하여 갖고 있는 훈련 데이터로부터 처음부터 임베딩 벡터를 훈련시키기도 하지만,
# 위키피디아 등의 방대한 데이터로 사전에 훈련된 워드 임베딩(pre-trained word embedding vector)를 가지고 와서 해당 벡터들의 값을 원하는 작업에
# 사용 할 수도 있습니다.

# 예를 들어서 감성 분류 작업을 하는데 훈련 데이터의 양이 부족한 상황이라면, 다른 방대한 데이터를 Word2Vec이나 GloVe 등으로 사전에 학습시켜놓은
# 임베딩 벡터들을 가지고 와서 모델의 입력으로 사용하는 것이 때로는 더 좋은 성능을 얻을 수 있습니다.
# 사전 훈련된 워드 임베딩을 가져와서 간단히 단어들의 유사도를 구해보는 실습을 해보겠습니다. 실제로 모델에 적용해보는 실습은 향후에 진행합니다.

# 구글이 제공하는 사전 훈련된(미리 학습되어져 있는) Word2Vec 모델을 사용하는 방법에 대해서 알아보겠습니다.
# 구글은 사전 훈련된 3백만 개의 Word2Vec 단어 벡터들을 제공합니다. 각 임베딩 벡터의 차원은 300입니다.
# gensim을 통해서 이 모델을 불러오는 건 매우 간단합니다. 이 모델을 다운로드하고 파일 경로를 기재하면 됩니다.

# 모델 다운로드 경로 : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

# 압축 파일의 용량은 약 1.5GB이지만, 파일의 압축을 풀면 약 3.3GB의 파일이 나옵니다.

import gensim
import urllib.request

# 구글의 사전 훈련된 Word2Vec 모델을 로드.
# 미리 다운로드 시켜놈
# urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", \
#                            filename="GoogleNews-vectors-negative300.bin.gz")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
# 모델의 크기(shape)를 확인해봅시다.

print(word2vec_model.vectors.shape)
# (3000000, 300)
# 모델의 크기는 3,000,000 x 300입니다. 즉, 3백만 개의 단어와 각 단어의 차원은 300입니다. 파일의 크기가 3기가가 넘는 이유를 계산해보면 아래와 같습니다.

# 3 million words * 300 features * 4bytes/feature = ~3.35GB
# 사전 훈련된 임베딩을 사용하여 두 단어의 유사도를 계산해봅시다.

print(word2vec_model.similarity('this', 'is'))
print(word2vec_model.similarity('post', 'book'))
# 0.407970363878
# 0.0572043891977
# 단어 'book'의 벡터를 출력해봅시다.

print(word2vec_model['book'])
# [ 0.11279297 -0.02612305 -0.04492188  0.06982422  0.140625    0.03039551
#  -0.04370117  0.24511719  0.08740234 -0.05053711  0.23144531 -0.07470703
# ... 300개의 값이 출력되는 관계로 중략 ...
#   0.03637695 -0.16796875 -0.01483154  0.09667969 -0.05761719 -0.00515747]
# 구현 링크 : https://github.com/nzw0301/keras-examples/blob/master/CBoW.ipynb

# 참고 : Word2vec 모델은 자연어 처리에서 단어를 밀집 벡터로 만들어주는 단어 임베딩 방법론이지만 최근에 들어서는 자연어 처리를 넘어서 추천 시스템에도
# 사용되고 있는 모델입니다. 적당하게 데이터를 나열해주면 Word2vec은 위치가 근접한 데이터를 유사도가 높은 벡터를 만들어준다는 점에서 착안된 아이디어입니다.

# 링크 : https://brunch.co.kr/@goodvc78/16?fbclid=IwAR1QZZAeZe_tNWxnxVCRwl8PIouBPAaqSIJ1lBxJ-EKtfDfmLehi1MUV_Lk

# 위 링크에서 그림 10번을 보면 쉽게 이해할 수 있습니다.