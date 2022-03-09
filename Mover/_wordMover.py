import numpy as np
import scipy as scp
import warnings
from nltk.tokenize import sent_tokenize

from mover._preprocessing import Preprocessor


def constraint_matrix_generator(n, m):
    
    """
        
        Constuct the contraint matrix for the LP defined by Word mover´s Distance
    
        params
        ------
        n : int -> number of distinct words in the first article
        m : int -> number of distinct words in the second article
    """
    
    A = np.zeros((n + m, n * m))
    
    mask_1 = np.ones((m,))
    
    mask_2 = np.array(([1] + [0] * (m-1)) * n)
    
    for i in range(n):
        A[i, i*m:i*m + m] = mask_1
        
    for i in range(n, n + m):
        
        A[i, :] = mask_2
        
        mask_2 = np.roll(mask_2, 1)
        
    return A
    
    
def _wmd(embedding_1, embedding_2, word_freq_1, word_freq_2, constraint_matrix=None):

    """
        Calculate the Word mover´s Distance
        
        parameters
        ---------
        - embedding_1 : np.array (distinct words x embedding dimensions) -> embedding of the first article
        - embedding_2 : np.array (distinct words x embedding dimensions) -> embedding of the second article
        - word_freq_1 : np.array -> frequency of each distinct word in article 1
        - word_freq_2 : np.array -> frequency of each distinct word in article 2
        - constraint_matrix : np.array -> see function constraint_matrix_generator
    
    """
    
    #
    # Normalize the word frequencies with respect to the L1 norm
    #
    
    word_freq_1 = word_freq_1 / np.linalg.norm(word_freq_1, 1)
    word_freq_2 = word_freq_2 / np.linalg.norm(word_freq_2, 1)
    

    #
    # Construction of the LP components
    #
    
    objective_coefficients = 1 - np.matmul(embedding_1, np.transpose(embedding_2, axes=(1, 0))).reshape((-1,))

    if constraint_matrix is None:
        constraint_matrix = constraint_matrix_generator(embedding_1.shape[0], embedding_2.shape[0])
    
    equality_constraints = np.concatenate([word_freq_1, word_freq_2])
    
    #
    # Invoking the scipy LP solver
    #
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scp.optimize.linprog(objective_coefficients,
                                       A_ub=None,
                                       b_ub=None,
                                       A_eq=constraint_matrix,
                                       b_eq=equality_constraints)
    except Exception as e:
        print(f'Exception occurred in the scipy LP solver. Exception: {e}')
        return False, 0
    
    #
    #   Clear up immediately
    #
    
    del objective_coefficients
    del equality_constraints
    
    # return state of the solver and the minimum objective function value
    return res.success, res.fun
    

def _sample(sample_distribution, sample_size):
    return np.random.choice(np.arange(sample_distribution.shape[0]),
                            sample_size,
                            replace=False,
                            p=sample_distribution)
        

def _monte_carlo_wmd(embedding_1,
                     embedding_2,
                     word_freq_1,
                     word_freq_2,
                     damping=0.2,
                     sims=5,
                     sample_size=30):

    """
        
        Monte Carlo type estimation of the Word mover´s Distance.
        For the embeddings of two articles the WMD is calculated on several samples of the words of the articles, with the words
        being sampled according to their TF-IDF score. The total WMD is calculated as average of the samples WMD.
        
        
        Rationale
        ---------
        The complexity of the WMD is quadratic in the sample size and even cubic in space complexity => we have a linear-quadratic trade-off

        parameters
        ----------
        - embedding_1 : np.array (distinct words x embedding dimensions) -> embedding of the first article
        - embedding_2 : np.array (distinct words x embedding dimensions) -> embedding of the second article
        - word_freq_1 : np.array -> frequency/score of each distinct word in article 1
        - word_freq_2 : np.array -> frequency/score of each distinct word in article 2
        - damping : float -> relaxes the softmax conversion of scores of words into a probability distribution
        - sims : int -> number of Monte Carlo simulations
        - sample_size : int -> maximum number of words drawn for each Monte Carlo run
        
        
        Returns
        -------
        success : bool -> true, if the WMD calculation for at least one sample was successful
        mean : float -> mean WMD across samples

    """

    # The constraint matrix can be precomputed and is fix
    constraint_matrix = constraint_matrix_generator(min(embedding_1.shape[0], sample_size),
                                                    min(embedding_2.shape[0], sample_size))

    # Convert word frequencies/td_idf sores (non-negative) into a probability distribution over words
    sample_distribution_1 = np.exp(damping * word_freq_1) / np.sum(np.exp(damping * word_freq_1))
    sample_distribution_2 = np.exp(damping * word_freq_2) / np.sum(np.exp(damping * word_freq_2))
    
    all_select_1 = np.arange(embedding_1.shape[0])
    all_select_2 = np.arange(embedding_2.shape[0])
    
    wmds = []
    for i in range(sims):
    
        if embedding_1.shape[0] <= sample_size:

            if embedding_2.shape[0] <= sample_size :
                #   One run is sufficient : no sampling required
                success, distance = _wmd(embedding_1,
                                         embedding_2,
                                         word_freq_1,
                                         word_freq_2,
                                         constraint_matrix)
                return success, distance

            #   Sample embedding_2
            selection_1 = all_select_1
            selection_2 = _sample(sample_distribution_2, sample_size)
            
        else:
            # Sample embedding_1
            selection_1 = _sample(sample_distribution_1, sample_size)
        
            if embedding_2.shape[0] > sample_size:
                #   Sample embedding_2
                selection_2 = _sample(sample_distribution_2, sample_size)
            else:
                selection_2 = all_select_2

        success, distance = _wmd(embedding_1[selection_1, :],
                                 embedding_2[selection_2, :],
                                 word_freq_1[selection_1],
                                 word_freq_2[selection_2],
                                 constraint_matrix)
                                
        if success:
            wmds.append(distance)
    
    if len(wmds) != 0:
        observations = np.array(wmds)
        return True, np.mean(observations)
        
    return False, None


def wmd(text1: str, text2: str, preprocessor_1: Preprocessor, preprocessor_2: Preprocessor):

    """
        Calculates Word Mover´s Distance using a Monte Carlo sampling approximation

        parameters
        ----------
        text1 : str
            First text of the pair to be compared
        text2 : str
            Second text of the pair to be compared
        preprocessor_1: mover._preprocessor.Preprocessor
            Configured preprocessor for argument text1
        preprocessor_2 : mover._preprocessor.Preprocessor
            Configured preprocessor for argument text2

        returns
        -------
        success : bool
            Flag indicating the success of the calculation
        wmd : float
            Monte Carlo approximated Word Mover´s Distance
    """

    embeddings_1, ((words_1, tfidf_1), valid_embeddings_1) = preprocessor_1.process(text1)
    embeddings_2, ((words_2, tfidf_2), valid_embeddings_2) = preprocessor_2.process(text2)

    success, distance = _monte_carlo_wmd(embeddings_1,
                                         embeddings_2,
                                         tfidf_1[valid_embeddings_1],
                                         tfidf_2[valid_embeddings_2],
                                         0.2,  # softmax damping for conversion of tf-idf
                                         10,  # number of monte carlo runs
                                         40)  # sample size per run
    return success, distance


def _norm_length(array):
    return (array / np.sqrt(np.sum(np.square(array))).reshape((-1, 1))).reshape((-1))


def _embed(sentences, model):
    norm = _norm_length(np.array([len(s) for s in sentences]))
    embeddings = model.encode(sentences)
    embeddings = embeddings / np.sqrt(np.sum(np.square(embeddings), axis=1)).reshape((-1, 1))
    return embeddings, norm


def smd(text1: str, text2: str, model):

    """
        Calculates Sentence Mover´s Distance using the passed SentenceTransformer.
        All texts are required to be in English.

        parameters
        ----------
        text1 : str
            First text of the pair to be compared
        text2 : str
            Second text of the pair to be compared
        model : SentenceTransformer

        returns
        -------
        success : bool
            Flag indicating the success of the calculation
        wmd : float
            Monte Carlo approximated Sentence Mover´s Distance

    """

    sentences_1 = sent_tokenize(text1)
    sentences_2 = sent_tokenize(text2)

    smd_embeddings_1, norm_1 = _embed(sentences_1, model)
    smd_embeddings_2, norm_2 = _embed(sentences_2, model)

    success, distance = _monte_carlo_wmd(smd_embeddings_1,
                                         smd_embeddings_2,
                                         norm_1,
                                         norm_2,
                                         0.2,  # softmax damping for conversion of tf-idf
                                         10,  # number of monte carlo runs
                                         40)  # sample size per run
    return success, distance
            
    
 
    
    
