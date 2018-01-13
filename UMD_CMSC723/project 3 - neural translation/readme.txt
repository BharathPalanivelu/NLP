This readme file is used to illustrate code about the transliteration from Chinese to English in Q9-Q11.

The following data files are the dataset without 'zhengma' decoding method: en_cn.train.txt, en_cn.val.txt, en_cn.test.txt

The following data files are the dataset with 'zhengma' decoding method: en_cnz.train.txt, en_cnz.val.txt, en_cnz.test.txt

Please use the following code for Chinese transliteration with 'zhengma' decoding method: python transliterate.py -t data/en_cnz.train.txt -v data/en_cnz.test.txt -n 20000 -o data/en_cnz.test.out >& log  

The true labels for the test set are stored into en_cnz.test.txt
The predicted labels for the test set are stored into en.test.out

The file 'crawler_and_datatransformation.py' is the script we used to collect the 'zhengma' code for the unique Chinese characters in our datasets and transform the datasets into the standard format. 

Here is the example output if running 2000 iterations

Reading lines in data/en_cnz.train.txt
Read 14465 word pairs
Vocabulary statistics
('cn', 40)
('en', 36)
1m 21s (- 1m 21s) (1000 50%) 1.8503
2m 45s (- 0m 0s) (2000 100%) 1.7267
Examples of output for a random sample of training examples
('INPUT: ', u'IDAIMBBDCFFZMBY')
('TARGET: ', u'katelinna')
('OUTPUT: ', u'kaiang<EOS>')

('INPUT: ', u'MDLNRS')
('TARGET: ', u'kandi')
('OUTPUT: ', u'kang<EOS>')

('INPUT: ', u'JAPYLK')
('TARGET: ', u'koupan')
('OUTPUT: ', u'kuang<EOS>')

('INPUT: ', u'WDEL')
('TARGET: ', u'kuan')
('OUTPUT: ', u'kuang<EOS>')

('INPUT: ', u'MFTEKMC')
('TARGET: ', u'kexing')
('OUTPUT: ', u'keng<EOS>')

Evaluate on unseen data
Read 1859 word pairs
Keeping 1859 word pairs for which all characters are in vocabulary
Average edit distance 6.2587
