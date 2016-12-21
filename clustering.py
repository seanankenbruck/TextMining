import numpy as np
import csv
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import scipy
import scipy.io as sio


######################## READ IN THE EMAILS TO PYTHON #############################
emails = pd.read_csv('C:/Users/TextMining/finishedFiles/test.csv')
#print emails

number = emails["EmailID"].tolist()
body = emails["Body"].tolist()
    
print len(body)

body2 = []
for i in body:
    printable = i.decode('ascii', 'ignore')
    body2.append(printable)
#print emailID[:10] #first 10 emails
#print email_body[2][:100]

# load nltk's English stopwords as variable called 'stopwords'
stop_words = nltk.corpus.stopwords.words('english')
stop_words.append(u'said')
stop_words.append(u"'s")
stop_words.append(u'wa')
stop_words.append(u'w')
stop_words.append(u'ha')
stop_words.append(u'thi')
stop_words.append(u'per')
stop_words.append(u'his')
stop_words.append(u'h')
stop_words.append(u'hillary')
stop_words.append(u'clintonemail.com')
stop_words.append(u'unclassified')
stop_words.append(u'date')
stop_words.append(u'has')
stop_words.append(u'this')
stop_words.append(u"'s")
stop_words.append(u'nan')
stop_words.append(u'h')


# load Porter's stemmer called 'PorterStemmer()'
porter = nltk.stem.porter.PorterStemmer()
# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    if tokens not in stop_words:
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [porter.stem(t) for t in filtered_tokens]
        return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in body2:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'body', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'

print vocab_frame.head()

#Tf-idf and document similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(body) #fit the vectorizer to email body

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

# KMEANS Clustering

from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# Create a data frame and map the clusters to the words
films = { 'ID': number, 'Text': body2, 'cluster': clusters }
frame = pd.DataFrame(films, index = [clusters] , columns = ['ID', 'Text', 'cluster'])
print frame['cluster'].value_counts()

# Top N words closest to the center of the

print "Top terms per cluster:"
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print "Cluster %d words:" % i
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print ' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0]
    print() #add whitespace
    print() #add whitespace
    
# Multidimensional Scaling
import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'arrive, depart, residence, conference, staff', 
                 1: 'democrat, civil servant, charge, art, christian', 
                 2: 'hait, haitian, earthquake, relief, million, hope', 
                 3: 'program, develop, fund, global, country, support', 
                 4: 'melanne verveer, woman, group, room'}
                 
#some ipython magic to show the matplotlib plots inline
%matplotlib inline 

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, text=body)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]["ID"], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
plt.savefig('money_clusters.png', dpi=200)


plt.close()

# Hierarchical document clustering
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200)
