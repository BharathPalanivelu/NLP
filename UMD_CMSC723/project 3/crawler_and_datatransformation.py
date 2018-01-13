import requests
from bs4 import BeautifulSoup as BS

url = 'http://www.cilin.org/dict/w_男.html'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}


## read existing data
# en_cn.train.txt
with open(r'data/en_cn.train.txt', 'r') as f:
    d1 = f.readlines()
# en_cn.val.txt 
with open(r'data/en_cn.val.txt', 'r') as f:
    d2 = f.readlines()
# en_cn.test.txt 
with open(r'data/en_cn.test.txt', 'r') as f:
    d3 = f.readlines()

d1_han = [x.split('\t')[1][:-1] for x in d1]
d2_han = [x.split('\t')[1][:-1] for x in d2]
d3_han = [x.split('\t')[1][:-1] for x in d3]

# find all unique characters
d_all = list(set(''.join(d1_han) + ''.join(d2_han) + ''.join(d3_han)))


## start the crawler
d_zheng_dict = {}
unable = 0

for count, i in enumerate(d_all):
    url_tmp = url.replace('男', i)
    try:
        d = requests.get(url_tmp, headers=headers)
        stat = BS(d.content, 'html.parser')
        for j in str(stat.find_all('p')[0]).split('<br/>'):
            if '郑码' in j:
                zheng = j.split('，')[0][3:]
                d_zheng_dict[i] = zheng
    except:
        unable += 1
    if count % 100 == 0:
        print('count {} of all {} words, unable to scrap {}'.format(count, len(d_all), unable))


## data transformation to standard format
d2_new = d1.copy()
d3_new = d2.copy()
d4_new = d3.copy()

# transform all chinese characters into zhengma
for i in range(len(d1_new)):
    tmp = d1_new[i].split('\t')
    tmp_han = tmp[1][:-1]
    tmp_han_new = []
    for j in tmp_han:
        try:
            tmp_zheng = d_zheng_dict[j]
            tmp_han_new.append(tmp_zheng)
        except:
            tmp_han_new.append(j)
    tmp[1] = ''.join(tmp_han_new) + '\n'
    d1_new[i] = tmp

for i in range(len(d2_new)):
    tmp = d2_new[i].split('\t')
    tmp_han = tmp[1][:-1]
    tmp_han_new = []
    for j in tmp_han:
        try:
            tmp_zheng = d_zheng_dict[j]
            tmp_han_new.append(tmp_zheng)
        except:
            tmp_han_new.append(j)
    tmp[1] = ''.join(tmp_han_new) + '\n'
    d2_new[i] = tmp

for i in range(len(d3_new)):
    tmp = d3_new[i].split('\t')
    tmp_han = tmp[1][:-1]
    tmp_han_new = []
    for j in tmp_han:
        try:
            tmp_zheng = d_zheng_dict[j]
            tmp_han_new.append(tmp_zheng)
        except:
            tmp_han_new.append(j)
    tmp[1] = ''.join(tmp_han_new) + '\n'
    d3_new[i] = tmp

d1_new = ['\t'.join(x) for x in d1_new]
d2_new = ['\t'.join(x) for x in d2_new]
d3_new = ['\t'.join(x) for x in d3_new]

train = ''.join(d1_new)
val = ''.join(d2_new)
test = ''.join(d3_new)


## output files
with open('data/en_cnz.train.txt', 'w') as f:
    f.write(train)
with open('data/en_cnz.val.txt', 'w') as f:
    f.write(val)
with open('data/en_cnz.test.txt', 'w') as f:
    f.write(test)