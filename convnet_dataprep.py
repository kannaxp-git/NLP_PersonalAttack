# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:20:11 2020

@author: kach
"""

from convokit import Corpus, download
corpus = Corpus(filename=download("conversations-gone-awry-corpus"))

corpus.print_summary_stats()

reviews = open("data/reviews.txt","w",encoding="utf-8")
label=open("data/labels.txt","w")

#i=0
for utt in corpus.iter_utterances():
    #i+=1
    txt=str(utt.text).replace('\n',' ')
    reviews.write(txt+'\n')
    if utt.meta['comment_has_personal_attack']:
        l='1'
    else:
        l='0'
    label.write(l+'\n')
    #if i>10:
    #    break

reviews.close()
label.close()