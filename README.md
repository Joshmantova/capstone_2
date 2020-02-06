# capstone_2
![](imgs/Top-Data-Science-Jobs.png)

# There are several specific types of data science jobs
## Can these types be depicted from the descriptions?
## Using clustering algorithms and natural langauge processing, we can answer this question

# The dataset:

# Data cleaning pipeline:
* Words that should not help differentiate the jobs from each other were identified (e.g. equal, disability)
** These were words that were included in just about every job

* Cleaning descriptions
** Remove punctuation using NLTK's RegexpTokenizer
** Remove stop words (including NLTK's default stop words and my identified stop words)
** Lemmatize each word using NLTK's WordNetLemmatizer
*** Reduces similar words (both semanticly and grammaticly related words) to a single word

![](imgs/wordcloud_only_punct_removed.png)

![](imgs/wordcloud_cleaned_descriptions.png)

* Transform the words into counts using SKLearn's count vectorizer and the tfidf vectorizer
** Some models require an unstandardized 


![](imgs/top_10_words_and_frequencies.png)

![](imgs/first_two_pca_dimensions.png)

