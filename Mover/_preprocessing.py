from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import Numberbatch as nb
import numpy as np


class Preprocessor :

    def __init__(self, language : str, language_abb : str) :
    
        """
            Params
            ------
            language : str -> language indentifier such as ('german')
            language_abb : str -> language abbreviation such as ('en')
        
        """
        
        self.language = language
        self.language_abb = language_abb
        
        
        self.stopwords = set(stopwords.words(language))
        self.punctuation = ['.','-','+',',',':','"','!','?',"'","''",'´´','``','`','´',"'s","n't",';','-','_']
        
        """
            Setup databases
        """
        
        self.f_database = nb.DatabaseConnection(nb.frequency_config)
        self.f_database.__enter__()
        
        self.e_database = nb.DatabaseConnection(nb.numberbatch_config)
        self.e_database.__enter__()
        
        
        """
            Preprocess registry
        """
        self.registry = {}
        self.registry['lower'] = self.lower
        self.registry['lowercasing'] = self.lower
        
        self.registry['punctuation'] = self.punctuation_removal
        
        self.registry['stopword'] = self.stopword_removal
        
        self.registry['tokenize'] = self.tokenize
        
        self.registry['tfidf_filter'] = self.tfidf_filter
        
        self.registry['embed'] = self.embed
        
    def config(self,processes : list) :
    
        self.pipeline = []
        
        for pro in processes :
            
            if type(pro) != list :
                raise Exception("A preprocess has to be configured as list of preprocess identifiers")
            
            if pro[0] not in self.registry :
                raise Exception(f'Unknown preprocess : {pro[0]}')
            
            #substitute preprocess identifier with preprocess function
            pro[0] = self.registry[pro[0]]
            
            if len(pro) == 1 :
                pro.append({}) #append an empty keyword argument dictionary
            
            self.pipeline.append(pro)
            
    def process(self,tokens = None) :
            
        additional_output = []
        for (pro,args) in self.pipeline :
        
            res = pro(tokens = tokens,**args)
            
            if type(res) == tuple :
                tokens = res[0]
                additional_output.append(res[1])
            else :
                tokens = res

        return tokens,additional_output
        
    def tokenize(self,tokens = None, **kwargs) :
    
        """
            NLTK tokenization of text
            
            Params
            ------
            tokens : str -> text to be tokenized
        """

        return word_tokenize(tokens)
         
    def lower(self,tokens = [],**kwargs) :
    
        """
            Lowercasing
        """
    
        return [token.lower() for token in tokens]
        
    def stopword_removal(self,tokens= [],**kwargs) :
    
        """
            
        """
        
        return [token for token in tokens if token not in self.stopwords]

    def punctuation_removal(self, tokens = [], **kwargs) :
    
        """
            
        """
        
        return [token for token in tokens if token not in self.punctuation]
        
    def tfidf_filter(self, tokens = [], **kwargs) :
    
        """
            Filters the tokens according to a tfidf threshold.
            Removes duplicate words.
            
            Keyword arguments
            -----------------
            - threshold : float -> tfidf threshold
            - frequency interpolation 
        
        """
    
        if 'threshold' not in kwargs :
            raise Exception("TF-IDF filter requires keyword argument threshold.")
        else :
            threshold = kwargs['threshold']
            
        metadata = 'frequency'
        
        if 'metadata' in kwargs :
        
            if kwargs['metadata'] in ['frequency','tf-idf'] :
                metadata = kwargs['metadata']
            else :
                raise Exception(f'metadata {kwargs["metadata"]} is not known. Use "frequency" or "tf-idf".')
                
        
        filtered_tokens = []
        token_count = {}
        
        #container for the token_counts ordered according to filtered_tokens
        counts = []   
        
        for token in tokens :
            
            if token not in token_count :
                token_count[token] = 1
            else :
                token_count[token] += 1
         
        distinct = list(token_count.keys())           
        
        frequencies = self.f_database.frequencies(self.language_abb,distinct)
                
        
        #
        #   Assign the average frequency substracted by one standard deviation frequency to an unknown word
        #
        
        if np.count_nonzero(np.isnan(frequencies)) != frequencies.shape[0] :
        
            np.nan_to_num(frequencies,copy=False,nan = np.nanmean(frequencies) - np.nanstd(frequencies))
            
        else : #no frequency available at all. Set default value
            
            frequencies = 0.05 * np.ones_like(frequencies)
        
        #
        #   Perform the tf-idf filtering
        #
                
        for i,token in enumerate(distinct) :

            tf_idf = token_count[token] * np.log(1 / (frequencies[i] + 1e-10))
                
            if tf_idf >= threshold :
                    
                filtered_tokens.append(token)
                
                if metadata == 'tf-idf' :
                    counts.append(tf_idf)   #use tf_idf instead of the token count
                elif metadata == 'frequency' :
                    counts.append(token_count[token])

                
            elif tf_idf is np.NaN :
                    
                pass
                
        return filtered_tokens,(filtered_tokens,np.array(counts))           
    
    def embed(self,tokens=[],**kwargs) :
    
        """
            ConceptNet Numberbatch embedding
        
        """
        
        embeddings = self.e_database.embed(self.language_abb,tokens)
            
        valid_embeddings = np.nonzero(~(np.isnan(embeddings))[:,0])[0]
        embeddings = embeddings[valid_embeddings,:]
        
        return embeddings,valid_embeddings
      