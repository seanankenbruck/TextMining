#Code to read a CSV file and produce the clusters using LDA
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import csv
from nltk.corpus import stopwords

#Read in the data from the CSV into Python
def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

emails = pd.read_csv('C:/Users/Sean Ankenbruck/Desktop/MSA/TextMining/finishedFiles/money.csv')
#print emails

number = emails["EmailID"].tolist()
body = emails["Body"].tolist()

print len(number)
print len(body)



############################# REMOVE STOP WORDS FROM TERM VECTORS ##############################

#Text Parsing: use NLTK to create the stopwords
stop_words = ['a','about','above','across','after','again','against','all','almost','alone','along','already','also','although','always','among','an','and','another','any','anybody','anyone','anything',
				'anywhere','are','area','areas','around','as','ask','asked','asking','asks','at','away','b','back','backed','backing','backs','be','became','because','become','becomes','been','before','began',
				'behind','being','beings','best','better','between','big','both','but','by','c','came','can','cannot','case','cases','certain','certainly','clear','clearly','come','could','d','did','differ','different',
				'differently','do','does','done','down','down','downed','downing','downs','during','e','each','early','either','end','ended','ending','ends','enough','even','evenly','ever','every','everybody','everyone',
				'everything','everywhere','f','face','faces','fact','facts','far','felt','few','find','finds','first','for','four','from','full','fully','further','furthered','furthering','furthers','g','gave','general',
				'generally','get','gets','give','given','gives','go','going','good','goods','got','great','greater','greatest','group','grouped','grouping','groups','h','had','has','have','having','he','her','here','herself',
				'high','high','high','higher','highest','him','himself','his','how','however','i','if','important','in','interest','interested','interesting','interests','into','is','it','its','itself','j','just','k','keep','keeps',
				'kind','knew','know','known','knows','l','large','largely','last','later','latest','least','less','let','lets','like','likely','long','longer','longest','m','made','make','making','man','many','may','me','member','members',
				'men','might','more','most','mostly','mr','mrs','much','must','my','myself','n','necessary','need','needed','needing','needs','never','new','new','newer','newest','next','no','nobody','non','noone','not','nothing','now',
				'nowhere','number','numbers','o','of','off','often','old','older','oldest','on','once','one','only','open','opened','opening','opens','or','order','ordered','ordering','orders','other','others','our','out','over','p','part',
				'parted','parting','parts','per','perhaps','place','places','point','pointed','pointing','points','possible','present','presented','presenting','presents','problem','problems','put','puts','q','quite','r','rather','really',
				'right','right','room','rooms','s','said','same','saw','say','says','second','seconds','see','seem','seemed','seeming','seems','sees','several','shall','she','should','show','showed','showing','shows','side','sides','since',
				'small','smaller','smallest','so','some','somebody','someone','something','somewhere','state','states','still','still','such','sure','t','take','taken','than','that','the','their','them','then','there','therefore','these','they',
				'thing','things','think','thinks','this','those','though','thought','thoughts','three','through','thus','to','today','together','too','took','toward','turn','turned','turning','turns','two','u','under','until','up','upon','us',
				'use','used','uses','v','very','w','want','wanted','wanting','wants','was','way','ways','we','well','wells','went','were','what','when','where','whether','which','while','who','whole','whose','why','will','with','within','without',
				'work','worked','working','works','would','x','y','year','years','yet','you','young','younger','youngest','your','yours','z','cheryl','pm']


############################ PORTER STEM REMAINING TERMS #################################

# Text Parsing: use NLTK to stem the text
porter = nltk.stem.porter.PorterStemmer()

#Define a function that will tokenize and stem the text using Porter
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [porter.stem(t) for t in filtered_tokens]
    return stems

#Define a function that will tokenize the text
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in body:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'email_body', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


#create a pandas DataFrame with the stemmed vocabulary as the index and the tokenized words as the column. 
#The benefit of this is it provides an efficient way to look up a stem and return a full token. 
#The downside here is that stems to tokens are one to many: the stem 'run' could be associated with 'ran', 'runs', 'running', etc. 
#For my purposes this is fine--I'm perfectly happy returning the first token associated with the stem I need to look up.
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print vocab_frame.head()

# STATUS: PASS

# ##############################TERM FREQUENCY-INVERSE DOCUMENT FREQUENCY #########################


#Create the TF-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=200000,
                        		min_df=0.10, stop_words='english',
                                use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
                                # from line 47

tfidf_matrix = tfidf_vectorizer.fit_transform(body) #fit the vectorizer to email body
print(tfidf_matrix.shape)
indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
features = tfidf_vectorizer.get_feature_names()
top_n = 30
top_features = [features[i] for i in indices[:top_n]]
print top_features

# STATUS: PASS

############################### KMEANS Cluster ########################################

# Part 3: CLUSTER - Kmeans is the easiest one
from sklearn.cluster import KMeans
terms = tfidf_vectorizer.get_feature_names()
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix) 
clusters = km.labels_.tolist()
films = { 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters] , columns = ['cluster'])
frame['cluster'].value_counts()

# STATUS: NO ERRORS - CONTINUE TO NEXT SECTION

# # ############################## LDA ########################################

# PART 4
#Running LDA - similar to LSA, based on probabilities
import string
from gensim import corpora, models, similarities 

#tokenize
tokenized_text = [tokenize_and_stem(text) for text in body]

#remove stop words
texts = [[word for word in text if word not in stop_words] for text in tokenized_text]

#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

from gensim import models 
lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

lda.show_topics()
topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix, dtype=object)
topic_words = topics_matrix[:,1]

for i in topic_words:
    print([[str(vocab_frame.loc[word[0]].ix[0,0])] for word in i])
    print()


############################## Words in each cluster #############################################
# Part 5 
docTopic = lda.get_document_topics(corpus,minimum_probability=0)
listDocProb = list(docTopic)

probMatrix = np.zeros(shape=(30215,10))
for i,x in enumerate(listDocProb):      #each document i
    for t in x:     #each topic j
        probMatrix[i, t[0]] = t[1] 

df = pd.DataFrame(probMatrix)

top_n = 1
topic_d = pd.DataFrame({n: df.T[col].nlargest(top_n).index.tolist() 
                  for n, col in enumerate(df.T)}).T
topic_d.columns = ['count']
topic_d['count'].value_counts()