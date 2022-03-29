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
# 1. 위키피디아로부터 데이터 다운로드 및 통합
# 위키피디아로부터 데이터를 파싱하기 위한 파이썬 패키지인 wikiextractor를 설치합니다.

# pip install wikiextractor
# 위키피디아 데이터를 다운로드 한 후에 전처리에서 사용할 형태소 분석기인 Mecab을 설치합니다.

# Colab에 Mecab 설치
# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
# !bash install_mecab-ko_on_colab190912.sh
# 위키피디아 덤프(위키피디아 데이터)를 다운로드합니다.

# !wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
# 위키익스트랙터를 사용하여 위키피디아 덤프를 파싱합니다.

# !python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2
# 현재 경로에 있는 디렉토리와 파일들의 리스트를 받아옵니다.

# %ls
# images/                                    LICENSE
# install_mecab-ko_on_colab190912.sh         README.md
# install_mecab-ko_on_colab_light_210108.sh  text/
# kowiki-latest-pages-articles.xml.bz2
# text라는 디렉토리 안에는 또 어떤 디렉토리들이 있는지 파이썬을 사용하여 확인해봅시다.

import os
import re
os.listdir('text')
# ['AG', 'AI', 'AH', 'AC', 'AE', 'AB', 'AA', 'AD', 'AF']
# AA라는 디렉토리의 파일들을 확인해봅시다.

# %ls text/AA
# wiki_00  wiki_12  wiki_24  wiki_36  wiki_48  wiki_60  wiki_72  wiki_84  wiki_96
# wiki_01  wiki_13  wiki_25  wiki_37  wiki_49  wiki_61  wiki_73  wiki_85  wiki_97
# wiki_02  wiki_14  wiki_26  wiki_38  wiki_50  wiki_62  wiki_74  wiki_86  wiki_98
# wiki_03  wiki_15  wiki_27  wiki_39  wiki_51  wiki_63  wiki_75  wiki_87  wiki_99
# wiki_04  wiki_16  wiki_28  wiki_40  wiki_52  wiki_64  wiki_76  wiki_88
# wiki_05  wiki_17  wiki_29  wiki_41  wiki_53  wiki_65  wiki_77  wiki_89
# wiki_06  wiki_18  wiki_30  wiki_42  wiki_54  wiki_66  wiki_78  wiki_90
# wiki_07  wiki_19  wiki_31  wiki_43  wiki_55  wiki_67  wiki_79  wiki_91
# wiki_08  wiki_20  wiki_32  wiki_44  wiki_56  wiki_68  wiki_80  wiki_92
# wiki_09  wiki_21  wiki_33  wiki_45  wiki_57  wiki_69  wiki_81  wiki_93
# wiki_10  wiki_22  wiki_34  wiki_46  wiki_58  wiki_70  wiki_82  wiki_94
# wiki_11  wiki_23  wiki_35  wiki_47  wiki_59  wiki_71  wiki_83  wiki_95
# 텍스트 파일로 변환된 위키피디아 한국어 덤프는 총 6개의 디렉토리로 구성되어져 있습니다. AA ~ AF의 디렉토리로 각 디렉토리 내에는
# 'wiki_00 ~ wiki_약 90내외의 숫자'의 파일들이 들어있습니다. 다시 말해 각 디렉토리에는 약 90여개의 파일들이 들어있습니다.
# 각 파일들을 열어보면 이와 같은 구성이 반복되고 있습니다.

# <doc id="문서 번호" url="실제 위키피디아 문서 주소" title="문서 제목">
# 내용
# </doc>
# 예를 들어서 AA 디렉토리의 wiki_00 파일을 읽어보면, 지미 카터에 대한 내용이 나옵니다.
#
# <doc id="5" url="https://ko.wikipedia.org/wiki?curid=5" title="지미 카터">
# 지미 카터
# 제임스 얼 "지미" 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39번째 대통령(1977년 ~ 1981년)이다.
# 지미 카터는 조지아 주 섬터 카운티 플레인스 마을에서 태어났다. 조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대
# 위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다.
# ... 이하 중략...
# </doc>
# 이제 이 6개 AA ~ AF 디렉토리 안의 wiki 숫자 형태의 수많은 파일들을 하나로 통합하는 과정을 진행해야 합니다. AA ~ AF 디렉토리 안의 모든 파일들의 경로를 파이썬의 리스트 형태로 저장합니다.

def list_wiki(dirname):
    filepaths = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        if os.path.isdir(filepath):
            # 재귀 함수
            filepaths.extend(list_wiki(filepath))
        else:
            find = re.findall(r"wiki_[0-9][0-9]", filepath)
            if 0 < len(find):
                filepaths.append(filepath)
    return sorted(filepaths)
filepaths = list_wiki('text')
# 총 파일의 개수를 확인해봅시다.

len(filepaths)
# 850
# 총 파일의 개수는 850개입니다. 이제 output_file.txt라는 파일에 850개의 파일을 전부 하나로 합칩니다.

with open("output_file.txt", "w") as outfile:
    for filename in filepaths:
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)
# 파일을 읽고 10줄만 출력해보겠습니다.

f = open('output_file.txt', encoding="utf8")

i = 0
while True:
    line = f.readline()
    if line != '\n':
        i = i+1
        print("%d번째 줄 :"%i + line)
    if i==10:
        break
f.close()
# 1번째 줄 :<doc id="5" url="https://ko.wikipedia.org/wiki?curid=5" title="지미 카터">

# 2번째 줄 :지미 카터

# 3번째 줄 :제임스 얼 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.

# 4번째 줄 :생애.

# 5번째 줄 :어린 시절.

# 6번째 줄 :지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.

# 7번째 줄 :조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다. 그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.

# 8번째 줄 :정계 입문.

# 9번째 줄 :1962년 조지아 주 상원 의원 선거에서 낙선하나 그 선거가 부정선거 였음을 입증하게 되어 당선되고, 1966년 조지아 주지사 선거에 낙선하지만, 1970년 조지아 주지사를 역임했다. 대통령이 되기 전 조지아주 상원의원을 두번 연임했으며, 1971년부터 1975년까지 조지아 지사로 근무했다. 조지아 주지사로 지내면서, 미국에 사는 흑인 등용법을 내세웠다.

# 10번째 줄 :대통령 재임.
# 2. 형태소 분석
from tqdm import tqdm
from konlpy.tag import Mecab
# 형태소 분석기 Mecab을 사용하여 토큰화를 진행해보겠습니다.

mecab = Mecab()
# 우선 output_file에는 총 몇 줄이 있는지 확인합니다.

f = open('output_file.txt', encoding="utf8")
lines = f.read().splitlines()
print(len(lines))
# 9718793
# 9,718,793개의 줄이 존재합니다. 상위 10개만 출력해봅시다.

print(lines[:10])
# ['<doc id="5" url="https://ko.wikipedia.org/wiki?curid=5" title="지미 카터">',
#  '지미 카터',
#  '',
#  '제임스 얼 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.',
#  '생애.',
#  '어린 시절.',
#  '지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.',
#  '조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다. 그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.',
#  '정계 입문.',
#  '1962년 조지아 주 상원 의원 선거에서 낙선하나 그 선거가 부정선거 였음을 입증하게 되어 당선되고, 1966년 조지아 주지사 선거에 낙선하지만, 1970년 조지아 주지사를 역임했다. 대통령이 되기 전 조지아주 상원의원을 두번 연임했으며, 1971년부터 1975년까지 조지아 지사로 근무했다. 조지아 주지사로 지내면서, 미국에 사는 흑인 등용법을 내세웠다.']
# 두번째 줄을 보면 아무런 단어도 들어있지 않은 ''와 같은 줄도 존재합니다. 해당 문자열은 형태소 분석에서 제외하도록 하고 형태소 분석을 수행해봅시다.

result = []

for line in tqdm(lines):
  # 빈 문자열이 아닌 경우에만 수행
  if line:
    result.append(mecab.morphs(line))
# 100%|██████████| 9718793/9718793 [15:27<00:00, 10478.61it/s]
# 빈 문자열은 제외하고 형태소 분석을 진행했습니다. 이제 몇 개의 줄. 즉, 몇 개의 문장이 존재하는지 확인해봅시다.

len(result)
# 6559314
# 6,559,314개로 문장의 수가 줄었습니다.

# 3. Word2Vec 학습
# 형태소 분석을 통해서 토큰화가 진행된 상태이므로 Word2Vec을 학습합니다.

from gensim.models import Word2Vec
model = Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)
model_result1 = model.wv.most_similar("대한민국")
print(model_result1)
# [('한국', 0.7382678389549255), ('미국', 0.6731516122817993), ('일본', 0.6541135907173157), ('부산', 0.5798133611679077), ('홍콩', 0.5752249360084534), ('서울', 0.5541036128997803), ('오스트레일리아', 0.5531408786773682), ('태국', 0.548468828201294), ('경상남도', 0.5462549924850464), ('제주특별자치도', 0.5385439395904541)]
model_result2 = model.wv.most_similar("어벤져스")
print(model_result2)
# [('스파이더맨', 0.80271977186203), ('트랜스포머', 0.773989200592041), ('아이언맨', 0.7648921012878418), ('스타트렉', 0.7645636796951294), ('어벤저스', 0.7626765966415405), ('엑스맨', 0.7586475610733032), ('《》,', 0.7560415267944336), ('트와일라잇', 0.7518032789230347), ('퍼니셔', 0.7391209602355957), ('테일즈', 0.7386105060577393)]
model_result3 = model.wv.most_similar("반도체")
print(model_result3)
# [('집적회로', 0.7714468836784363), ('연료전지', 0.7699108719825745), ('전자', 0.7606919407844543), ('웨이퍼', 0.745188295841217), ('실리콘', 0.743209958076477), ('트랜지스터', 0.7398351430892944), ('PCB', 0.7275883555412292), ('TSMC', 0.7156406044960022), ('가속기', 0.6962155699729919), ('광전자', 0.6957612037658691)]
