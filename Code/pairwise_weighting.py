import numpy as np

"""
I'm not aware of many pairwise weighting schemes that exist for tree taxa. One could imagine calculating
the weights of each member of a pair by methods described in "weighting_methods.py" and combining them.

Thus, this file currently contains a method that I made up and that is all.

Possible to do list:
    1. Write method to calculate pairwise weights from dictionary of single terminal weights
    2. Figure out how to code the ACL pairwise method
"""



def get_weight_matrices(my_tree):
    """
    This is code that implements a method I developed to get weights for pairwise comparisons
    that is similar in ideology to the GSC (Gerstein, Sonnhammer, Chothia) algorithm for weighting
    individual nodes in a tree. I wouldn't be surprised if someone else came up with this idea at 
    some point but I've yet to see it. At the end of the GSC method, the weights for all terminals
    is equivalent to the total branch length. Which is to say the branch length of the tree gets 
    pushed/pulled to the terminals in a weighted fashion. A similar process occurs here where one
    can envision laying out all pairwise lineages and performing calculations on this set of lines
    where each individual segment of any given line contributes a weight inversely proportional to 
    how many lines that the segment appears in. A line of branch length 5 that appears in 100 pairs
    contributes 5/100 of a weight to each pair. A line of branch length 5 that apperas in 20 pairs
    contributes 5/20 to each of the pairs, etc. At the end each gets added up to give a weight for
    the overall pairwise comparison.

    Note that this calculation may alter the original tree by first setting the root node to have a 
    length of zero and then making it equal to None at the end.

    Input(s):
    my_tree - a Bio.Phylo tree object. No root required!

    Output(s):
    ending_weight_matrix - the matrix of weights for all terminal-terminal pairs (rows and columns)
    ending_weight_matrix_normalized - ending_weight_matrix divided by the ending_matrix_raw. Sort of 
            a %-uniqueness score for each pairwise comparison
    ending_matrix_raw - the matrix of distances between all terminal-terminal pairs
    my_tree.get_terminals() - list of terminals that are critical for knowing row/col entries in 
            the matrices. This should be obvious but am returning it to be super sure everyone knows

    """
    #Instantiate the matrices that I'll use throughout
    term_count = len(my_tree.get_terminals())
    starting_weight_matrix = np.zeros((term_count, term_count))
    starting_matrix_raw = np.zeros((term_count, term_count))
    
    #Set root branch length to zero
    my_tree.root.branch_length = 0.
    term_pairs = enumerate(my_tree.get_terminals())
    term_pairs = [(pair[1], pair[0]) for pair in term_pairs]
    term_dict = dict(term_pairs)
    
    ending_weight_matrix, ending_matrix_raw, finished\
            = pairwise_distances_recursive(my_tree.root, my_tree,\
                                           starting_weight_matrix,\
                                           starting_matrix_raw,\
                                           term_dict, [])
    ending_weight_matrix_normalized = np.divide(ending_weight_matrix, ending_matrix_raw,\
                                     out=np.zeros_like(ending_weight_matrix), where=ending_matrix_raw!=0)
    #Reset the root branch length 
    my_tree.root.branch_length = None
    return ending_weight_matrix, ending_weight_matrix_normalized, ending_matrix_raw, my_tree.get_terminals()

def pairwise_distances_recursive(putative_root, my_tree, weight_matrix, matrix_raw, term_map_dict, finished):
    """
    Side note, this took weeks, WEEKS, to get working and I'm very proud of it's overall simplicity despite the 
    handful of notes and the obvious copy/pasting that shouldn't be there.

    Input(s):
    putative_root - a Bio.Phylo clade object, should probably be my_tree.root to start the process
    my_tree - the Bio.Phylo tree
    weight_matrix and matrix_raw - start as n x n np.zeros matrices where n is the number of terminals 
            in the tree. These matrices get values added to them throughout the recursion
    term_map_dict - a dictionary that speeds things up by indexing key:val pairs 
            as: "terminal object":"index in matrices"
    finished - an empty list to start and tracks progress of the recursive algorithm
    
    Output(s):
    weight_matrix - the matrix of weights for all terminal-terminal pairs (rows and columns)
    matrix_raw - the matrix of distances between all terminal-terminal pairs
    finished - a list of visited nodes for record keeping

    """
    ###Speeding things up by passing ds_terms and us_terms around a bit
    ds_terms = [term_map_dict[i] for i in putative_root.get_terminals()]
    us_terms = [term_map_dict[i] for i in my_tree.get_terminals() if term_map_dict[i] not in ds_terms] 
    
    #This initial "if" really just skips the root which should have no upstream terms and the branch length
    #along the root node is contained in Bio.Phylo trees by the branch_length of the two descendants
    if min([len(ds_terms), len(us_terms)]) > 0:
        ###################
        ###For the weights matrix
        ###################
        val = putative_root.branch_length/(len(ds_terms)*len(us_terms))
        #Start at zero
        to_add = np.zeros(weight_matrix.shape)
        #Add rows and column values
        to_add[len(finished):len(ds_terms)+len(finished), :] = val        
        to_add[:, len(finished):len(ds_terms)+len(finished)] = val
        #Subtract center square
        to_add[len(finished):len(ds_terms)+len(finished), len(finished):len(ds_terms)+len(finished)] = 0
        #Add to matrix
        weight_matrix += to_add
        
        #####################
        ###For the raw matrix
        #####################
        #Which is  to say, partitioning the branch_length to all straddling terminals
        #without dividing it by the number of straddling terminals
        val = putative_root.branch_length
        #Start at zero
        to_add = np.zeros(matrix_raw.shape)
        #Add rows and column values
        to_add[len(finished):len(ds_terms)+len(finished), :] = val        
        to_add[:, len(finished):len(ds_terms)+len(finished)] = val
        #Subtract center square
        to_add[len(finished):len(ds_terms)+len(finished), len(finished):len(ds_terms)+len(finished)] = 0
        #Add to matrix
        matrix_raw += to_add
        
    if len(putative_root.clades) == 2:
        weight_matrix, matrix_raw, finished = pairwise_distances_recursive(putative_root.clades[0],\
                                                                      my_tree, weight_matrix, matrix_raw,\
                                                                      term_map_dict, finished)
        weight_matrix, matrix_raw, finished = pairwise_distances_recursive(putative_root.clades[1],\
                                                                      my_tree, weight_matrix, matrix_raw,\
                                                                      term_map_dict, finished)
    elif len(putative_root.clades) == 0:
        finished.append(putative_root)
    return weight_matrix, matrix_raw, finished





