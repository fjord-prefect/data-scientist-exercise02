import re

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

import spacy
nlp = spacy.load("en_core_web_sm")

class top_words:
    def __init__(self, df):
        #using narrative to get broader vocabulary for Count Vectorizer
        #less missing values for narrative
        init_doc = list(df['narrative'].apply(lambda x:self.pre_process(x)))
        
        #create a vocabulary of words, 
        #ignore words that appear in 85% of documents, 
        #eliminate stop words
        cv=CountVectorizer(max_df=0.85,stop_words=set(stopwords.words('english')),ngram_range=(1,2))
        cv.fit(init_doc)
        
        self.cv = cv
        
    def remove_stops(self, string_):
        stop_words = set(stopwords.words('english')) 

        word_tokens = word_tokenize(string_) 

        filtered_sentence = [w for w in word_tokens if not w in stop_words] 

        return ' '.join(filtered_sentence)

    def pre_process(self, text):

        # lowercase
        text=text.lower()

        #remove tags
        text=re.sub("","",text)

        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)

        return text

    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        #use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:

            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]

        return results

    def get_top_words(self, df, text_column, mask, limit):
        
        #preprocess based on mask and intended text column
        docs = df[mask][text_column].apply(lambda x:self.pre_process(x))
        
        word_count_vector = self.cv.transform(docs)
        feature_names = self.cv.get_feature_names()
        
        #init and fit tfidf transformer
        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(word_count_vector)
        
        #generate tf-idf for the given document
        tf_idf_vector = tfidf_transformer.transform(self.cv.transform(docs))

        #sort the tf-idf vectors by descending order of scores
        sorted_items=self.sort_coo(tf_idf_vector.tocoo())

        #extract only the top n; n here is 10
        keywords=self.extract_topn_from_vector(feature_names,sorted_items,limit)

        return keywords