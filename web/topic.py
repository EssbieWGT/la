#borrows heavily from here: https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985

#run as:

# packages to store and manipulate data
import pandas as pd
import numpy as np

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

# model building package
import sklearn

# package to clean text
import texthero as hero
from texthero import stopwords
import re

#########DEFINE FUNCTIONS#########

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

#load tweets
tweets = pd.read_csv("./TwitClips/allNH.csv")

# make a new column to highlight retweets
tweets['is_retweet'] = tweets['content'].apply(lambda x: x[:2]=='RT')
tweets['is_retweet'].sum()  # number of retweets

# 10 most repeated tweets
tweets.groupby(['content']).size().reset_index(name='counts')\
  .sort_values('counts', ascending=False).head(10)

# make new columns for mentioned usernames and hashtags
tweets['mentioned'] = tweets.content.apply(find_mentioned)
tweets['hashtags'] = tweets.content.apply(find_hashtags)

######HASHTAG ANALYSIS#############

#Note... you couldn't get this section to work. IT as just coming back "NaN" even though you know there's got to be SOME correlation here.
#COME BACK TO THIS LATER. Try looking at the full code for the blog post.

# number of unique hashtags
# flattened_hashtags_df = pd.DataFrame(tweets['hashtags'].apply(pd.Series).stack().reset_index(drop=True), columns=["hashtag"])
# flattened_hashtags_df.unique().size
#
# # make new dataframe
# hashtag_vector_df = flattened_hashtags_df.loc[:,['hashtag']]
# for hashtag in flattened_hashtags_df['hashtag']:
#     # make columns to encode presence of hashtags
#     hashtag_vector_df['{}'.format(hashtag)] = hashtag_vector_df.apply(
#         lambda hashtag_list: int(hashtag in hashtag_list))

#############TOPIC MODELING SECTION#############

#Note: You use TextHero to do most of the cleaning

def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'(@[A-Za-z0–9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?', "", elem))
    return df

#note, you're not sure if you should pull out these custom stop words just yet, but are going for it anyway
default_stopwords = stopwords.DEFAULT
custom_stopwords = default_stopwords.union(set(["heroes","nursing","home","homes","amp"]))

tweets["clean_tweet"] = tweets["content"]
tweets = clean_text(tweets,"clean_tweet")

tweets['clean_tweet'] = (
    tweets['clean_tweet']
    .pipe(hero.remove_urls)
    .pipe(hero.clean)
    .pipe(hero.remove_stopwords)
)

tweets['clean_tweet']= hero.remove_stopwords(tweets['clean_tweet'],custom_stopwords)

from sklearn.feature_extraction.text import CountVectorizer
# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(max_df=0.9, min_df=100, token_pattern='\w+|\$[\d\.]+|\S+')
# apply transformation
tf = vectorizer.fit_transform(tweets['clean_tweet']) #.toarray()
# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()
tf.shape # --> (200000, 2296)

from sklearn.decomposition import LatentDirichletAllocation
number_of_topics = 10
model = LatentDirichletAllocation(n_components=number_of_topics, random_state=45) # random state for reproducibility
# Fit data to model
model.fit(tf)

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


#GAVE UP AND WENT IN A DIFFERENT DIRECTION:

import gensim
from spacy.tokenizer import Tokenizer
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

#Visualizations
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls

df = tweets

df['toke'] = (
    df['clean_tweet']
    .pipe(hero.tokenize)
)

#taken from here https://towardsdatascience.com/twitter-topic-modeling-e0e3315b12e2

id2word = Dictionary(df['toke'])
print(len(id2word))

id2word.filter_extremes(no_below=2, no_above=.99)
print(len(id2word))

corpus = [id2word.doc2bow(d) for d in df['toke']]

# Instantiating a Base LDA model
base_model = LdaMulticore(corpus=corpus, num_topics=10, id2word=id2word, workers=12, passes=10)

# Filtering for words
words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics()]

# Create Topics
topics = [' '.join(t[0:10]) for t in words]

# Getting the topics
for id, t in enumerate(topics):
    print(f"------ Topic {id} ------")
    print(t, end="\n\n")


# Compute Perplexity
# a measure of how good the model is. lower the better
base_perplexity = base_model.log_perplexity(corpus)
print('\nPerplexity: ', base_perplexity)

# Compute Coherence Score
coherence_model = CoherenceModel(model=base_model, texts=df['toke'],
                                   dictionary=id2word, coherence='c_v')
coherence_lda_model_base = coherence_model.get_coherence()
print('\nCoherence Score: ', coherence_lda_model_base)


#Creating Topic Distance Visualization
visualisation = pyLDAvis.gensim_models.prepare(base_model, corpus, id2word)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

#Defining a function to loop over number of topics to be used to find an
#optimal number of tipics
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics
#
#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics
#
#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the
#     LDA model with respective number of topics
#     """
#     coherence_values_topic = []
#     model_list_topic = []
#     for num_topics in range(start, limit, step):
#         model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list_topic.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values_topic.append(coherencemodel.get_coherence())
#     return model_list_topic, coherence_values_topic
#
# # Can take a long time to run.
# model_list_topic, coherence_values_topic = compute_coherence_values(dictionary=id2word,
#                                                         corpus=corpus,
#                                                         texts=df['toke'],
#                                                         start=2, limit=200, step=6)

#try and predict

base_model.get_term_topics("cuomo")

from gensim import corpora, models
from gensim.corpora import Dictionary

# get_document_topics for a document with a single token 'user'
dictionary = corpora.Dictionary(df['toke'])
# text = df['toke'][1]
text = ["nursing","homes","are","infrastructure"]
bow = dictionary.doc2bow(text)
t = base_model.get_document_topics(bow)
print(base_model.get_document_topics(bow))

topic=[]
for i in df['toke']:
    text = i
    bow = dictionary.doc2bow(text)
    t = base_model.get_document_topics(bow)
    try:
        topic.append(t[0][0])
    except:
        topic.append("0")

df["topic"] = topic

cvs=[]
for j in df['topic']:
    if j == 8:
        cvs.append("Y")
    else:
        cvs.append("N")

df["F"]=cvs
justCVS = df[df["F"]=="Y"]
len(justCVS)

justCVS.content.sample(10).values

df.["topic"].str.find(1)

####################################
#scanning clips for mentions of research
import pandas as pd
import re

df = pd.read_csv("/Users/essbie/Downloads/la2.csv")

terms = ["research","survey","new report"]

# df2 = df[df['text'].str.findall("survey", flags=re.IGNORECASE)]

rm = []
for i in df["text"]:
    try:
        if any(x in i for x in ["research","survey","new report"]) is True:
            rm.append("Yes")
        else:
            rm.append("No")
    except:
        rm.append("No")

df["found"]=rm
research = df[df['found']=="Yes"]

research.to_csv("ra2.csv")
