from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import mover._numberbatch as nb
import numpy as np


class Preprocessor:

    def __init__(self, language: str, language_abb: str):
    
        """
            Preprocessor pipeline class.

            parameters
            ------
            language : str -> language indentifier such as ('german')
            language_abb : str -> language abbreviation such as ('en')
        
        """
        
        self.language = language
        self.language_abb = language_abb

        self.stopwords = set(stopwords.words(language))
        self.punctuation = ['.', '-', '+', ',', ':', '"', '!', '?', "'",
                            "''", '´´', '``', '`', '´', "'s",
                            "n't", ';', '-', '_']
        
        # set up databases
        self.f_database = nb.DatabaseConnection(nb.frequency_config)
        self.f_database.__enter__()
        
        self.e_database = nb.DatabaseConnection(nb.numberbatch_config)
        self.e_database.__enter__()

        self.pipeline = None

        # preprocess registry
        self.registry = dict()
        self.registry['lower'] = self._lower
        self.registry['lowercasing'] = self._lower
        self.registry['punctuation'] = self._punctuation_removal
        self.registry['stopword'] = self._stopword_removal
        self.registry['tokenize'] = self._tokenize
        self.registry['tfidf_filter'] = self._tfidf_filter
        self.registry['embed'] = self._embed

    def config(self, processes: list):
        """
            Configures preprocessing pipeline.

            parameters
            ----------
            processes : list
                List of string identifiers of preprocessing functions registered in the class

        """
        self.pipeline = []
        
        for pro in processes:
            
            if type(pro) != list:
                raise Exception("A preprocess has to be configured as list of preprocess identifiers")
            
            if pro[0] not in self.registry :
                raise Exception(f'Unknown preprocess : {pro[0]}')
            
            # substitute preprocess identifier with preprocess function
            pro[0] = self.registry[pro[0]]
            
            if len(pro) == 1:
                # append an empty keyword argument dictionary
                pro.append({})
            
            self.pipeline.append(pro)
            
    def process(self, tokens=None):

        """
            Applies preprocesses on the passed text

            parameters
            ----------
            tokens : str
                Text to be preprocessed

        """
        if self.pipeline is None:
            raise ReferenceError("The pipeline has not been configures.")

        additional_output = []
        for (pro, args) in self.pipeline:
        
            res = pro(tokens=tokens, **args)
            
            if type(res) == tuple:
                tokens = res[0]
                additional_output.append(res[1])
            else:
                tokens = res

        return tokens, additional_output
        
    def _tokenize(self, tokens=None, **kwargs):
        return word_tokenize(tokens)
         
    def _lower(self, tokens = [], **kwargs):
        return [token.lower() for token in tokens]
        
    def _stopword_removal(self, tokens= [], **kwargs):
        return [token for token in tokens if token not in self.stopwords]

    def _punctuation_removal(self, tokens=[], **kwargs):
        return [token for token in tokens if token not in self.punctuation]
        
    def _tfidf_filter(self, tokens=[], **kwargs):
    
        if 'threshold' not in kwargs:
            raise Exception("TF-IDF filter requires keyword argument threshold.")
        else:
            threshold = kwargs['threshold']
            
        metadata = 'frequency'
        
        if 'metadata' in kwargs:
        
            if kwargs['metadata'] in ['frequency', 'tf-idf']:
                metadata = kwargs['metadata']
            else:
                raise Exception(f'metadata {kwargs["metadata"]} is not known. Use "frequency" or "tf-idf".')

        filtered_tokens = []
        token_count = {}
        
        #container for the token_counts ordered according to filtered_tokens
        counts = []   
        
        for token in tokens:
            
            if token not in token_count:
                token_count[token] = 1
            else :
                token_count[token] += 1
         
        distinct = list(token_count.keys())           
        
        frequencies = self.f_database.frequencies(self.language_abb,distinct)

        #  Assign the average frequency subtracted by one standard deviation frequency to an unknown word
        if np.count_nonzero(np.isnan(frequencies)) != frequencies.shape[0]:
            np.nan_to_num(frequencies,
                          copy=False,
                          nan=np.nanmean(frequencies) - np.nanstd(frequencies))
            
        else: #no frequency available at all. Set default value
            frequencies = 0.05 * np.ones_like(frequencies)
        
        # perform filtering
        for i, token in enumerate(distinct):
            tf_idf = token_count[token] * np.log(1 / (frequencies[i] + 1e-10))
            if tf_idf >= threshold:
                filtered_tokens.append(token)
                if metadata == 'tf-idf':
                    counts.append(tf_idf)   #use tf_idf instead of the token count
                elif metadata == 'frequency':
                    counts.append(token_count[token])
            elif tf_idf is np.NaN:
                pass
                
        return filtered_tokens, (filtered_tokens, np.array(counts))
    
    def _embed(self, tokens=[], **kwargs):

        embeddings = self.e_database.embed(self.language_abb, tokens)
            
        valid_embeddings = np.nonzero(~(np.isnan(embeddings))[:, 0])[0]
        embeddings = embeddings[valid_embeddings, :]
        
        return embeddings, valid_embeddings
      