import numpy as np
import scipy as scp

def constraint_matrix_generator(n,m) :
    
    """
        
        Constuct the contraint matrix for the LP defined by Word Mover´s Distance
    
        params
        ------
        n : int -> number of distinct words in the first article
        m : int -> number of distinct words in the second article
    """
    
    A = np.zeros((n + m,n * m))
    
    mask_1 = np.ones((m,))
    
    mask_2 = np.array(([1] + [0] * (m-1)) * n)
    
    for i in range(n) :
        A[i,i * m : i * m + m] = mask_1
        
    for i in range(n,n + m) :
        
        A[i,:] = mask_2
        
        mask_2 = np.roll(mask_2,1)
        
    return A
    
    
def wmd(embedding_1, embedding_2, word_freq_1, word_freq_2, constraint_matrix = None) :

    """
        Calculate the Word Mover´s Distance 
        
        Params
        ------
        - embedding_1 : np.array (distinct words x embedding dimensions) -> embedding of the first article
        - embedding_2 : np.array (distinct words x embedding dimensions) -> embedding of the second article
        - word_freq_1 : np.array -> frequency of each distinct word in article 1
        - word_freq_2 : np.array -> frequency of each distinct word in article 2
        - constraint_matrix : np.array -> see function constraint_matrix_generator
    
    """
    
    #
    # Normalize the word frequencies with respect to the L1 norm
    #
    
    word_freq_1 = word_freq_1 / np.linalg.norm(word_freq_1,1)
    word_freq_2 = word_freq_2 / np.linalg.norm(word_freq_2,1)
    

    #
    # Construction of the LP components
    #
    
    objective_coefficients = 1 - np.matmul(embedding_1,np.transpose(embedding_2,axes=(1,0))).reshape((-1,))

    if constraint_matrix is None :
        constraint_matrix = constraint_matrix_generator(embedding_1.shape[0],embedding_2.shape[0])
    
    equality_constraints = np.concatenate([word_freq_1,word_freq_2])
    
    #
    # Invoking the scipy LP solver
    #
    
    try :
        res = scp.optimize.linprog(objective_coefficients,
                                        A_ub=None,
                                        b_ub=None,
                                        A_eq=constraint_matrix,
                                        b_eq=equality_constraints
                                        #bounds=[(0,1)] * objective_coefficients.shape[0]  ->per default non-negative
                                    )
    except Exception as e:
        print(f'Exception occurred in the scipy LP solver. Exception: {e}')
        return False,0
    
    #
    #   Clear up immediately
    #
    
    del objective_coefficients
    del equality_constraints
    
    # return state of the solver and the minimum objective function value
    return res.success,res.fun
    
    
    
def sample(sample_distribution, sample_size) :

    """
        
        Sample embedding vectors according to the the score in frequencies
    
    """
    
    return np.random.choice(np.arange(sample_distribution.shape[0]),sample_size,replace=False,p=sample_distribution)
        

def monte_carlo_wmd(embedding_1,embedding_2, word_freq_1, word_freq_2, damping = 0.2, sims = 5, sample_size = 30) :

    """
        
        Monte Carlo type estimation of the Word Mover´s Distance.
        For the embeddings of two articles the WMD is calculated on several samples of the words of the articles, with the words
        being sampled according to their TF-IDF score. The total WMD is calculated as average of the samples WMD.
        
        
        Rationale
        ---------
        The complexity of the WMD is quadratic in the sample size and even cubic in space complexity => we have a linear-quadratic trade-off
        
        
        Params
        ------
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
        std : float -> standard deviation of the WMD across samples
    
    """


    #
    # The constraint matrix can be precomputed and is fix
    #
    
    constraint_matrix = constraint_matrix_generator(min(embedding_1.shape[0],sample_size),
                                                    min(embedding_2.shape[0],sample_size))
                                                    
      
    #
    # Convert word frequencies/td_idf sores (non-negative) into a probability distribution over words
    #
    
    sample_distribution_1 = np.exp(damping * word_freq_1) / np.sum(np.exp(damping * word_freq_1))
    sample_distribution_2 = np.exp(damping * word_freq_2) / np.sum(np.exp(damping * word_freq_2))
    
    all_select_1 = np.arange(embedding_1.shape[0])
    all_select_2 = np.arange(embedding_2.shape[0])
    
    wmds = []

    for i in range(sims) :
    
        if embedding_1.shape[0] <= sample_size :
        
        
            if embedding_2.shape[0] <= sample_size :
            
                #
                #   One run is sufficient : no sampling required
                #
            
                success,distance = wmd(embedding_1,embedding_2, word_freq_1, word_freq_2,constraint_matrix)
            
                return success,distance,0
            
            
        
            #
            #   Sample embedding_2
            #
            
            selection_1 = all_select_1
            selection_2 = sample(sample_distribution_2,sample_size)
            
        else :
        
            #
            # Sample embedding_1
            #
    
            selection_1 = sample(sample_distribution_1,sample_size)
        
            if embedding_2.shape[0] > sample_size :
            
                #
                #   Sample embedding_2
                #
            
            
                selection_2 = sample(sample_distribution_2,sample_size)
              
            else :
        
                selection_2 = all_select_2
              
           
        success,distance = wmd( embedding_1[selection_1,:],
                                embedding_2[selection_2,:],
                                word_freq_1[selection_1],
                                word_freq_2[selection_2],
                                constraint_matrix)
                                
        if success :
                    
            wmds.append(distance)
    
    if len(wmds) != 0 :
    
        observations = np.array(wmds)
        
        return True,np.mean(observations)
        
    return False,None
            
            
    
 
    
    
