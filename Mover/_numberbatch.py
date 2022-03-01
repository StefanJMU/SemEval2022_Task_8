import psycopg2
import numpy as np

numberbatch_config = {
  'user': 'postgres',
  'password': 'root',
  'host': 'localhost',
  'port' : 5432,
  'database': 'Numberbatch'
}

frequency_config = {

  'user': 'postgres',
  'password': 'root',
  'host': 'localhost',
  'port' : 5432,
  'database': 'Frequency'
}


#Numberbatch and Wordlex use sometimes different language identifiers.
#-> Artifact from database construction
lang_identifier_map = {

    'en' : 'eng',
    'fr' : 'fre',
    'ar' : 'ara',
    'tr' : 'tur'
}

def for_database(database_name) :

    def wrapper(function) :
    
        def func(s,*args, **kwargs) :
            
            if s.config['database'] == database_name :
                return function(s,*args,**kwargs)
            else :
                print(f'Database {s.config["database"]} has no function {function}')
        return func
        
    return wrapper

class DatabaseConnection :
    
    def __init__(self, config) :    
        
        self.config = config
        
        #Prepared statements for database construction
        
        self.insert_prep = ("INSERT INTO vectors (lang,concept,embedding) VALUES (%s,%s,%s);")
        self.insert_frequency_prep = ("INSERT INTO frequency (index,lang,word,freq) VALUES (%s,%s,%s,%s);")
        
        self.make_partition_prep = ("CREATE TABLE {}_{} PARTITION OF {} FOR VALUES IN ('{}');")
        self.make_index_prep = ("CREATE INDEX ON {}_{} ({});")
        
        self.get_embedding_prep = ("SELECT embedding FROM vectors where lang='{}' and concept='{}'")
        
        self.get_frequency_prep = ("SELECT freq FROM frequency where lang='{}' and word='{}'")
        
        self.commit_counter = 0
      
    def __enter__(self) :
        self.cnx = psycopg2.connect(**self.config) 
        self.cursor = self.cnx.cursor()
        
        return self
        
    def __exit__(self, type, value, traceback) :
        self.cursor.close()
        self.cnx.close()
    
    def make_partition(self, table,lang,index_key) :
        
        """
            Create new partition in the database for a new language
        """
        self.cursor.execute(self.make_partition_prep.format(table,lang,table,lang))
        
        self.cursor.execute(self.make_index_prep.format(table,lang,index_key))
        
        self.cnx.commit()
    
    
    @for_database('Numberbatch')
    def insert_concept(self, lang, concept, embedding_vector) :
        
        """
            Insert concept into the embedding database
            
            Params
            -----
            lang : str -> language abbreviation ('de' etc.)
            concept : str -> term
            embedding_vector : str with structre '{float,...,float}'
        """
        self.cursor.execute(self.insert_prep,(lang,concept,embedding_vector))
        
        self.commit_counter += 1
        
        if self.commit_counter % 100 == 0 :
            self.cnx.commit()
            
    @for_database('Frequency')
    def insert_frequency(self, index,lang, word, freq) :
    
        """
            Insert concept into the embedding database
            
            Params
            -----
            lang : str -> language abbreviation ('de' etc.)
            word : str -> term
            freq : float -> frequency of the word
        """
        
        self.cursor.execute(self.insert_frequency_prep,(index,lang,word,freq))
        
        self.commit_counter += 1
        
        if self.commit_counter % 100 == 0 :
            self.cnx.commit()
        
          
    @for_database('Numberbatch')
    def get_embedding(self, lang, concept) :
        
        """
            Retrieve embedding vector for concept. Specification of the language is
            required due to overlapping of words in languages and for efficient access via
            the PostGreSQL partition concept
        
        """
        
        try :
            self.cursor.execute(self.get_embedding_prep.format(lang,concept))
        
            res = self.cursor.fetchone()
            
            if res is not None :
                return np.array(res[0])  #list of projected attributes returned
            
        except :
            
            #print("Error during embedding retrieval of concept {} of language {}. Returning NaN embedding.".format(concept,lang))
            pass
                
        res = np.empty((300,))
        res[:] = np.NaN
        
        return res
        
    @for_database('Frequency')
    def get_frequency(self,lang,concept) :
    
        """ 
            Get the document frequency for concept with respect to the most frequent word of lang, according to WordLex
        """
        
        try :
        
            self.cursor.execute(self.get_frequency_prep.format(lang,concept))
        
            res = self.cursor.fetchone()
        
            if res is not None :
                return res[0]  #list of projected attributes returned
                
        except :
                        
            #
            #   Commit the current queries. After an error has been thrown the database will reject every further request until commitment
            #
            
            self.cnx.commit()
            
        return np.NaN
        
    @for_database('Numberbatch')
    def embed(self, lang, concepts) :
        
        """
            TODO: A caching strategy can be used here as well
            
        """ 
        
        if len(concepts) == 0 :
            raise Exception("Function embed received empty list of concepts")
            
        vectors = []
        for concept in concepts :
            
            vectors.append(self.get_embedding(lang,concept))
            
        self.cnx.commit()
        
        return np.stack(vectors,axis=0)
           
    @for_database('Frequency')
    def frequencies(self, lang, concepts) :
    
        """
            WordLex uses sometimes different other language abbreviations than Numberbatch
            TODO: Synchronize the databases in this regard
        """
        
        if lang in lang_identifier_map :
            lang = lang_identifier_map[lang]
            
        freqs = []
        for concept in concepts :
            freqs.append(self.get_frequency(lang,concept))
            
        self.cnx.commit()
            
        return np.array(freqs)
        
        
        
        
        
        
        