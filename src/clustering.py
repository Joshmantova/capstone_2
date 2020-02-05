import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from sklearn.preprocessing import normalize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.stem import SnowballStemmer

import pyLDAvis
import pyLDAvis.gensim


df = pd.read_csv('../Datasets/df_all_linkedin.csv', index_col=0)
df_co = pd.read_csv('../Datasets/df_linkedin_Colorado.csv', index_col=0)

descriptions = df['Description'].values
descriptions_co = df_co['Description'].values

stopWords = set(stopwords.words('english'))
add_stopwords = {
    'join', 'work', 'team', 'future', 'digital', 'technology', 'access', 'leader', 'industry', 'history', 'innovation',
    'year', 'customer', 'focused', 'leading', 'business', 'ability', 'country', 'employee', 'www', 'seeking',
    'location', 'role', 'responsible', 'designing', 'code', 'ideal', 'candidate', 'also', 'duty', 'without', 'excellent',
    'set', 'area', 'well', 'use', 'strong', 'self', 'help', 'diverse', 'every', 'day', 'equal', 'employment', 'opportunity',
    'affirmative', 'action', 'employer', 'diversity', 'qualified', 'applicant', 'receive', 'consideration', 'regard',
    'race', 'color', 'religion', 'sex', 'national', 'origin', 'status', 'age', 'sexual', 'orientation', 'gender',
    'identity', 'disability', 'marital', 'family', 'medical', 'protected', 'veteran', 'reasonable', 'accomodation',
    'protect', 'status', 'equal', 'discriminate', 'inclusive', 'diverse'
}
stopWords = stopWords.union(add_stopwords)

tokenize_remove_punct = RegexpTokenizer(r'\w+')
lemma = WordNetLemmatizer()

def remove_stopwords(stopWords, descriptions):
    cleaned_descriptions = []
    for description in descriptions:
        temp_list = []
        for word in description.split():
            if word not in stopWords:
                temp_list.append(word.lower())
        cleaned_descriptions.append(' '.join(temp_list))
    return np.array(cleaned_descriptions)

def remove_punctuation(reg_tokenizer, descriptions):
    no_punct_descriptions = []
    for description in descriptions:
        description_no_punct = ' '.join(reg_tokenizer.tokenize(description))
        no_punct_descriptions.append(description_no_punct)
    return np.array(no_punct_descriptions)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
               'N': wordnet.NOUN,
               'V': wordnet.VERB,
               'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_descriptions(lemmatizer, descriptions):
    cleaned_descriptions = []
    for description in descriptions:
        temp_list = []
        for word in description.split():
            cleaned_word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            temp_list.append(cleaned_word)
        cleaned_descriptions.append(' '.join(temp_list))
    return np.array(cleaned_descriptions)

descriptions_no_sw_co = remove_stopwords(stopWords, descriptions_co)
descriptions_no_sw_punct_co = remove_punctuation(tokenize_remove_punct, descriptions_no_sw_co)
cleaned_descriptions_co = lemmatize_descriptions(lemma, descriptions_no_sw_punct_co)

descriptions_no_sw = remove_stopwords(stopWords, descriptions)
descriptions_no_sw_punct = remove_punctuation(tokenize_remove_punct, descriptions_no_sw)
cleaned_descriptions = lemmatize_descriptions(lemma, descriptions_no_sw_punct)

vectorizer = CountVectorizer(stop_words=stopWords, min_df=.15, max_df=0.75, max_features=5000)
tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords, min_df=.15, max_df=0.75, max_features=5000)
tfidf = tfidf_vectorizer.fit_transform(cleaned_descriptions).toarray()
tf = vectorizer.fit_transform(cleaned_descriptions)

kmeans = KMeans(n_clusters=5, verbose=True, n_jobs=-1)
kmeans.fit(tfidf)

sorted_centroids = []
for cluster in kmeans.cluster_centers_:
    top_10 = np.argsort(cluster)[::-1]
    sorted_centroids.append(top_10[:10])


for idx, c in enumerate(sorted_centroids): 
    print(f'\nCluster {idx}\n')
    for idx in c: 
        print(tfidf_vectorizer.get_feature_names()[idx]) 

silhouette_score(tfidf, kmeans.labels_)
kmeans.score(tfidf)

feature_names = vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_components=4, 
                                max_iter=10, learning_method='online', 
                                random_state=0, verbose=True, n_jobs=-1)

lda.fit(tf)

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print('Topic %d:' % (topic_idx))
        print(' '.join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))
        
        
num_top_words=10
display_topics(lda, feature_names, num_top_words)
data_text = df[['Description']].copy()
data_text['index'] = data_text.index
documents = data_text
def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, get_wordnet_pos(text))
#     return stemmer.stem(WordNetLemmatizer().lemmatize(text, get_wordnet_pos(text)))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stopWords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

stemmer = SnowballStemmer('english')
processed_docs = documents['Description'].map(preprocess)

id2word = gensim.corpora.Dictionary(processed_docs)
id2word.filter_extremes(no_below=80, no_above=.75, keep_n=5000)
texts = processed_docs
bow_corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=3, id2word=id2word, passes=10, random_state=0)
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, id2word)
vis