# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.decomposition import PCA

plt.style.use('fivethirtyeight')

# Loading in the data
df = pd.read_csv('../Datasets/df_all_linkedin.csv')
descriptions = df['Description'].values

# Updating NLTK's stopwords with one's I've identified from this dataset
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
stopWords_full = stopWords.union(add_stopwords)

# Initializing the punctuation remover and lemmatizer
tokenize_remove_punct = RegexpTokenizer(r'\w+')
lemma = WordNetLemmatizer()

def remove_stopwords(stopWords, descriptions):
    cleaned_descriptions = []
    for description in descriptions:
        temp_list = []
        for word in description.split():
            if word not in stopWords_full:
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

# Cleaning the descriptions
descriptions_no_punct = remove_punctuation(tokenize_remove_punct, descriptions)
descriptions_no_punct_sw = remove_stopwords(stopWords_full, descriptions_no_punct)
cleaned_descriptions(lemmatize_descriptions(lemma, descriptions_no_punct_sw))

# Vectorizing cleaned descriptions and creating TF matrix
vectorizer = CountVectorizer(stop_words='english')
tf = vectorizer.fit_transform(descriptions_no_sw_punct)

# Calculating feature frequencies
feature_names = vectorizer.get_feature_names()
feature_frequencies = np.sum(tf.toarray(), axis=0)

# Getting the indecies of the most frequent words
order = np.argsort(feature_frequencies)[::-1]
for o in order[:10]:
    print(feature_names[o], feature_frequencies[o])

top_10_words = []
top_10_word_freqs = []
for o in order[:10]:
    top_10_words.append(feature_names[o])
    top_10_word_freqs.append(feature_frequencies[o])
    
fig, ax = plt.subplots()
ax.barh(top_10_words[::-1], top_10_word_freqs[::-1])
ax.set_ylabel('Top 10 Words')
ax.set_xlabel('Frequencies')
ax.set_title('Top 10 Words and Their Frequencies')
plt.tight_layout()

# Word cloud with only punctuation removed.

no_punct_descriptions = remove_punctuation(tokenize_remove_punct, descriptions)
flattened_descriptions = ''
for description in no_punct_descriptions:
    flattened_descriptions += (description.lower() + ' ')


wordcloud = WordCloud(width=800, height=800,
                     background_color='white',
                     min_font_size=10).generate(flattened_descriptions)

fig, ax = plt.subplots()
ax.imshow(wordcloud)
plt.axis('off')
plt.tight_layout()

flat_cleaned_descriptions = ' '
for descrip in cleaned_descriptions:
    flat_cleaned_descriptions += (descrip + ' ')

wordcloud_cleaned = WordCloud(width=800, height=800,
                             background_color='white',
                             min_font_size=10).generate(flat_cleaned_descriptions)

fig, ax = plt.subplots()
ax.imshow(wordcloud_cleaned)
plt.axis('off')
plt.tight_layout()

tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords_full, max_features=50000)
tfidf = tfidf_vectorizer.fit_transform(cleaned_descriptions).toarray()

pca = PCA(n_components=2, random_state=0)
pca_tfidf = pca.fit_transform(tfidf)

var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(pca_tfidf[:, 0], pca_tfidf[:, 1],
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First two PCA dimensions")
ax.set_xlabel("1st eigenvector (PC1)")
ax.set_ylabel("2nd eigenvector (PC2)");

