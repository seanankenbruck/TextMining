# Sentiment Analysis
from anew_module import anew
# term = 'happy'
# print anew.exist( term )
# print anew.sentiment( term )
import csv
import re
import string 
import pandas as pd
#Read in the data from the CSV into Python
emails = pd.read_csv('C:/Users/Sean Ankenbruck/Desktop/MSA/TextMining/finishedFiles/money.csv')
#print emails

number = emails["EmailID"].tolist()
body = emails["Body"].tolist()
email_id = []
#end: Read in the data from the CSV into Python


# STATUS: PASS
from nltk.tokenize import sent_tokenize as sent_tokenize
from nltk.tokenize import word_tokenize as word_tokenize

#tokenize into sentences. 
for i in range(0,len(body)):
    body[i] = sent_tokenize(body[i])

#create new list of sentences in order of speech (not nested lists).
sentence_list=[]
for i in range(0,len(body)):
    for j in range(0,len(body[i])):
        sentence_list.append(body[i][j])
        email_id.append(number[j])

      
#remove punctuation, change to lower-case, and tokenize to word level
punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
term_vec = [ ]

for d in sentence_list:
    d = d.lower()
    d = punc.sub( '', d )
    term_vec.append( word_tokenize( d ) )


print len(term_vec)

# term_list = term_vec[100]
# print anew.exist(term_list)
# print anew.sentiment(term_list)
valence=[]
arousal=[]
sentiment_list=[]


for i in term_vec:
    if sum(anew.exist(i)) >=2:
        sentiment_list.append(i)
        valence.append ((anew.sentiment(i))['valence'])
        arousal.append ((anew.sentiment(i))['arousal'])

values = zip(sentiment_list, valence, arousal, email_id)

print ('Writing file')
headers = ['sentiment list','valence', 'arousal', 'ID']
with open('email_scores.csv', 'wb') as out:
    writer = csv.writer( out )
    #write headers
    writer.writerow(headers)
    #write the csv file
    for value in values:
        writer.writerow(value)

# STATUS: PASSED JUST NEED EMAIL IDS! 