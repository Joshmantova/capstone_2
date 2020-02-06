# capstone_2
![](imgs/Top-Data-Science-Jobs.png)

# There are several specific types of data science jobs
## Can these types be depicted from the descriptions?
## Using clustering algorithms and natural langauge processing, we can answer this question

# The dataset:

# Data cleaning pipeline:
![](imgs/capstone_2_flowchart.png)

* Words that should not help differentiate the jobs from each other were identified (e.g. equal, disability)
    * These were words that were included in just about every job

* Cleaning descriptions
    * Remove punctuation using NLTK's RegexpTokenizer
    * Remove stop words (including NLTK's default stop words and my identified stop words)
    * Lemmatize each word using NLTK's WordNetLemmatizer
        * Reduces similar words (both semanticly and grammaticly related words) to a single word

## Words with only punctuation removed
![](imgs/wordcloud_only_punct_removed.png)

## Words after cleaning pipeline
![](imgs/wordcloud_cleaned_descriptions.png)

* Transform the words into counts using SKLearn's count vectorizer and the tfidf vectorizer
    * Some models require an unstandardized count of each word
    * Other models require a count standardized by how often that word occurs in the corpus

    * Removed words that occured in less than 15% of documents
    * Also removed words that occured in more than 75% of documents
    * Only kept 5000 of the most frequent words

## Top 10 words and their frequencies
![](imgs/top_10_words_and_frequencies.png)


## Visualizing the descriptions after vectorizing in two dimensions
![](imgs/first_two_pca_dimensions.png)
* Was hoping to identify a certain number of clusters
    * Clusters don't seem to be clear in only two dimensions

## Perhaps k-means clustering can help identify the proper number of clusters to choose
* Modeled the data using K-Means clustering with a varying number of clusters
![](imgs/silhouette_scores_k_values.png)

* Hoped to see a clear peak of this graph, but it appears that the model just keeps getting better and better as number of clusters increases
* Point of this project would be lost if too many clusters were extracted

## Given this data, I changed my approach from trying to numerically figure out what the best number of clusters. Instead, I decided to run several models and choose the number of clusters and type of model that seems to yield the most useful clusters.

K-Means Clustering:
Cluster 0

ancestry
record
million
applicable
look
create
law
challenge
change
design

Cluster 1

product
analytics
model
analysis
insight
skill
company
management
marketing
team

Cluster 2

learn
machine
system
model
research
software
scientist
engineering
program
engineer

Read more [here](../notebooks/lda3.html)
