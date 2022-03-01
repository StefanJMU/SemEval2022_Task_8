from Mover import monte_carlo_wmd, Preprocessor
from sentence_transformers import SentenceTransformer


en_preprocessor = Preprocessor('english','en')

en_preprocessor.config([
    ['tokenize'],
    ['lower'],
    ['punctuation'],
    ['stopword'],
    ['tfidf_filter',{'threshold' : 5,'metadata' : 'tf-idf'}],  
    ['embed']    
])

text1 = "Stock markets crashed on monday."
text2 = "The weather is expected to be nice on monday."


#
# Word Mover´s Distance
#

embeddings_1,((words_1,tfidf_1),valid_embeddings_1) = en_preprocessor.process(text1)
embeddings_2,((words_2,tfidf_2),valid_embeddings_2) = en_preprocessor.process(text2)

success,distance = monte_carlo_wmd(embeddings_1,
                               embeddings_2,
                               tfidf_1[valid_embeddings_1],
                               tfidf_2[valid_embeddings_2],
                               0.2,#softmax damping for conversion of tf-idf 
                               10, #number of monte carlo runs
                               40) #sample size per run
   
if success :   
    print(f'Distance : {distance}')
else :
    print("Fail.")


#
# Sentence Mover´s Distance
#


def norm_length(array) :

  return (array / np.sqrt(np.sum(np.square(array))).reshape((-1,1))).reshape((-1))

def embed(sentences,model) :

  norm = norm_length(np.array([len(s) for s in sentences]))
  embeddings = model.encode(sentences)
  embeddings = embeddings / np.sqrt(np.sum(np.square(embeddings),axis=1)).reshape((-1,1))

  return embeddings,norm
  
model = SentenceTransformer('all-MiniLM-L12-v2')

smd_embeddings_1, norm_1 = embed(sentences, model)
smd_embeddings_2, norm_2 = embed(sentences_2, model)
   
success,distance = monte_carlo_wmd(smd_embeddings_1,
                                   smd_embeddings_2,
                                   norm_1,
                                   norm_2,
                                   0.2,#softmax damping for conversion of tf-idf 
                                   10, #number of monte carlo runs
                                   40) #sample size per run
                       
if success :   
    print(f'Distance : {distance}')
else :
    print("Fail.")
