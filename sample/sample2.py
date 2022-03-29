"""
단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 FastText가 있습니다.
Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있습니다.
Word2Vec와 FastText와의 가장 큰 차이점이라면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다.
내부 단어. 즉, 서브워드(subword)를 고려하여 학습합니다.
"""
import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

# from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models import KeyedVectors

# 1) 훈련 데이터 이해하기
# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

# 2) 훈련 데이터 전처리하기
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]
print('총 샘플의 개수 : {}'.format(len(result)))

# 샘플 3개만 출력
for line in result[:3]:
    print(line)

# 3) Word2Vec 훈련시키기
# from gensim.models import Word2Vec
# from gensim.models import KeyedVectors

model = FastText(sentences=result, size=100, window=5, min_count=5, workers=4, sg=1)
# FastText의 하이퍼파라미터값은 다음과 같습니다.

# size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
# window = 컨텍스트 윈도우 크기
# min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
# workers = 학습을 위한 프로세스 수
# sg = 0은 CBOW, 1은 Skip-gram.

# Word2Vec에 대해서 학습을 진행하였습니다. Word2Vec는 입력한 단어에 대해서 가장 유사한 단어들을 출력하는 model.wv.most_similar을 지원합니다.
# man과 가장 유사한 단어들은 어떤 단어들일까요?
model_result = model.wv.most_similar("man")
print(model_result)

# 4) Word2Vec 모델 저장하고 로드하기
model.wv.save_word2vec_format('eng_ft') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_ft") # 모델 로드
# 로드한 모델에 대해서 다시 man과 유사한 단어를 출력해보겠습니다.

model_result = loaded_model.most_similar("man")
print(model_result)
