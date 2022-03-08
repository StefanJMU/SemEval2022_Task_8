from mover import wmd, smd, Preprocessor
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
# Word mover´s Distance
#
success, distance = wmd(text1,
                        text2,
                        en_preprocessor,
                        en_preprocessor)

if success :   
    print(f'Distance : {distance}')
else :
    print("Fail.")


#
# Sentence mover´s Distance
#

model = SentenceTransformer('all-MiniLM-L12-v2')
success, distance = smd(text1, text2, model)
                       
if success :   
    print(f'Distance : {distance}')
else :
    print("Fail.")
