from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer 

import numpy as np
import re
import os

class SearchEngine101:
    def __init__(self, dir):
        all_files = os.listdir(dir)
        # print(all_files)
        self.text_content= [open(file_name, 'rb').read() for file_name in all_files]
        self.preprocess()
        # print(type(self.cleaned_text))
        # split_text = [self.tokenize(text) for text in self.cleaned_text]
        # print(split_text)
        self.index()
        print ("Preprocessing and Indexing done...")

    def preprocess(self):
        """
        Cleans self.texts and sets the cleaned_texts attribute.
        """

        self.cleaned_text =[]
        self.all_letter =[]
        # print(self.text_content[:0], '\n\n\n\n\n', self.text_content[:1], '\n\n\n\n\n', self.text_content[:2])
        for text in self.text_content:
            self.all_letter.append(re.sub(b'[^a-zA-Z ]',b'',text))
            # self.cleaned_text.append(self.all_letter.lower())

        for text in self.all_letter:
            self.cleaned_text.append(text.lower())

        '''
        for text in self.cleaned_t:
            self.cleaned_text =text.decode()
        print(type(self.cleaned_text))
        '''

    def tokenize(self, text):
        # print(type(text))
        # words =text.split(' ')
        words =text.split(b' ')
        # print(words)
        # !!! remove stopwords

        # Follow https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
        non_stop_words =[word for word in words if word not in stopwords.words('english')]
        filtered_object =filter(None, non_stop_words)
        filtered_list =list(filtered_object)
        # print('the filtered list is here! ')
        # print(filtered_list)
        return filtered_list

    def index(self):
        split_texts = [self.tokenize(text) for text in self.cleaned_text]
        # We'll use a dictionary (HashMap) for storing the inverted index
        # Key : word
        # Value : ???
        self.inverted_index = {}
        # print(split_texts)
        for index,split_text in enumerate(split_texts):
            split_text = set(split_text)
            for word in split_text:
                self.inverted_index[word] = self.inverted_index.get(word, [])
                self.inverted_index[word].append(index)

        sorted(self.inverted_index.items(), key =lambda x: x[1])
        # print(self.inverted_index)

    def intersection(self, list1, list2):
        """
        Should return intersection of list1 and list2
        :param list1: list of integers
        :param list2: list of integers
        :return:
        """
        # intersection = []
        # !!! populate the intersection list and return
        intersection = list(set(list1)&set(list2))
        return intersection

    def filter(self, query):
        """
        Returns the filtered list of texts [both cleaned and original] which contain the query terms
        :param query: string - user query
        :return: filterd_list: list - list of documents that contain the query terms
        """
        in_query =str.encode(query)
        query_terms = self.tokenize(in_query)
        # Retrieve ??? for each of the terms in the 
        # print(query_terms)
        document_lists = [self.inverted_index[term] for term in query_terms]
        # !!!Optimise the document lists for faster intersection

        # Now, iteratively take intersection
        document_indices = document_lists[0]
        for document_list in document_lists[1:]:
            document_indices = self.intersection(document_indices, document_list)

        return [self.cleaned_text[index] for index in document_indices], [self.text_content[index] for index in document_indices]


    def vectorize(self, filtered_texts, query):
        """
        Store the vectors and vectorizer.
        """
        
        # !!! Use TfIdfVectorizer. It automatically converts into a matrix.
        self.vectors = None
        self.vectorizer = TfidfVectorizer(min_df =1)
        in_query =str.encode(query)
        query_list =self.tokenize(in_query)
        # print(query_list)

        # print(np.shape(filtered_texts),'\n\n\n\n\n\n\n\n')
        # print(filtered_texts)
        text_matrix =self.vectorizer.fit_transform(filtered_texts)
        #print(query_list)
        query_matrix =self.vectorizer.fit_transform(query_list)
        # print(query_matrix)

        self.vectors =(text_matrix, query_matrix)

    def retrieve_ranked_list(self):
        """
        Return the indices of text_vectors in decreasing order of cosine similarity and the scores
        :param text_vectors: the vectors of the text
        :param query_vector: the vector of the query
        :return: indices, scores: indices of top 10 documents and scores of all documents
        """
        similarities = []
        
        # !!! Populate the similarities array with cosine similarities between text_vectors and query_vector
        '''
        print(self.vectors[0])
        print(np.shape(self.vectors[0]), '\n\n\n\n\n\n\n', ) 
        print(np.shape(self.vectors[1]))
        '''
        sparse_mat =self.vectors[0].getrow(0).todense()
        # print(self.vectors[1])
        sparse_mat_y =self.vectors[1].getrow(0).todense()
        #print(sparse_mat_y)
        mat_x =sparse_mat.transpose()
        y =list(np.array(sparse_mat_y).reshape(-1,).tolist())

        # y =[y for x in Y for y in x]
        # print('y is:',y)
        # print(np.shape(mat_x)[:1])
        r =np.shape(mat_x)[:1]
        rang =r[0]
        #print(rang)
        # print(type(mat_x[0]))
        # a =list(np.array(mat_x[0]).reshape(-1,).tolist())
        for i in range(rang):
            j =i+1
            if (j+1 >rang):
                break
            a =list(np.array(mat_x[i]).reshape(-1,).tolist())
            b =list(np.array(mat_x[j]).reshape(-1,).tolist())
            if(a[0] ==0 and b[0] ==0):
                # print('was I  here? ')
                continue

            l=[a,b]
            # print(l)
            flat_l =[y for x in  l for y in x]
            np.reshape(flat_l, 2)
            #m np.reshape(flat_ly, 2)
            
            X =list(map(int, flat_l))
            Y =list(map(int, y))
            # print(X, Y)
            # print(flat_l, y)
            
            
            similarity =cosine(flat_l, y)
            similarities.append(similarity)
            
            # print(similaritie,'\n')
            # similarities.append(similaritie)
            
            
            
            

        # print(X)
        # transf_X =np.delete(X, np.s_[:-1])
        # print(transf_X)
        # print(np.reshape(X, (2, 92)))
        # similarities =cosine_similarity(sparse_mat, sparse_mat_y)
        # Return the top 10 indices of the similarities array
        return np.argsort(similarities)[::-1][:10], similarities

    def print_list(self,text_content, scores, text_indices):

        print (len(text_indices), "Results Found!\n")
        print ("*******************************\n")
        for index in text_indices:
            print (scores[index])
            # print (text_content[index])
            print ("*******************************\n")

    def search(self, query):
        filtered_clean, filtered_orig = self.filter(query)
        self.vectorize(filtered_texts=filtered_clean, query=query)
        text_indices, scores = self.retrieve_ranked_list()
        # print(filtered_clean[1])
        self.print_list(text_content=filtered_clean, scores= scores, text_indices=text_indices)
        
    def analyse(self):
        """
        This is for your custom code. Use it to analyse IDF/TF-IDF and construct queries.
        :return:

        """
        pass

  
if __name__ == "__main__":
    # Write the driver code here
    engine = SearchEngine101(dir = 'data_raw')
    # cleaned_texts =engine.cleaned_texts
    # split_texts = [engine.tokenize() for text in cleaned_texts]
    # print(cleaned_texts)
    # print(split_texts)
    # engine.preprocess()
    # print(engine.index())
    query ='dalhousie university'
    # in_query =str.encode(query)
    engine.search(query)


    



