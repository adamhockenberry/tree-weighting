import numpy as np


def get_weight_matrices(my_tree):
    term_count = len(my_tree.get_terminals())
    starting_matrix_weighted = np.zeros((term_count, term_count))
    starting_matrix_raw = np.zeros((term_count, term_count))
    my_tree.root.branch_length = 0.
    term_pairs = enumerate(my_tree.get_terminals())
    term_pairs = [(pair[1], pair[0]) for pair in term_pairs]
    term_dict = dict(term_pairs)
    
    ending_matrix_weighted, ending_matrix_raw, finished\
            = pairwise_distances_recursive(my_tree.root, my_tree,\
                                                              starting_matrix_weighted,\
                                                              starting_matrix_raw,\
                                                              term_dict, [])
    ending_matrix_normalized = np.divide(ending_matrix_weighted, ending_matrix_raw,\
                                     out=np.zeros_like(ending_matrix_weighted), where=ending_matrix_raw!=0)
    my_tree.root.branch_length = None
    return ending_matrix_raw, ending_matrix_weighted, ending_matrix_normalized

def pairwise_distances_recursive(putative_root, my_tree, matrix_weighted, matrix_raw, term_map_dict, finished):
    '''
    my_tree is a Bio.Phylo tree
    putative_root should probably be my_tree.root to start the process
    matrix_weighted and matrix_raw are nxn np.zeros matrices where n is the number of terminals in the tree
    term_map_dict speeds life up by indexing key:val pairs as: "terminal object":"index in matrices"
    finished is just an empty list to start and tracks progress of the recursive algorithm

    Side note, this took weeks, WEEKS, to get working and I'm very proud of it's overall simplicity despite the 
    handful of notes and the obvious copy/paste that shouldn't be there.
    '''
    ###Could speed things up by passing ds_terms and us_terms around a bit
    ds_terms = [term_map_dict[i] for i in putative_root.get_terminals()]
    ###Should make this a set addition problem at very least
    us_terms = [term_map_dict[i] for i in my_tree.get_terminals() if term_map_dict[i] not in ds_terms] 
    if min([len(ds_terms), len(us_terms)]) > 0:
        ###################
        ###For the weighted matrix
        ###################
        val = putative_root.branch_length/(len(ds_terms)*len(us_terms))
        ###Start at zero
        to_add = np.zeros(matrix_weighted.shape)
        ###Add rows and column values
        to_add[len(finished):len(ds_terms)+len(finished), :] = val        
        to_add[:, len(finished):len(ds_terms)+len(finished)] = val
        ###Subtract center square
        to_add[len(finished):len(ds_terms)+len(finished), len(finished):len(ds_terms)+len(finished)] = 0
        ###Add to matrix
        matrix_weighted += to_add
        
        #####################
        ###For the raw matrix (copy/paste of above, basically)
        ###################
        val = putative_root.branch_length
        ###Start at zero
        to_add = np.zeros(matrix_raw.shape)
        ###Add rows and column values
        to_add[len(finished):len(ds_terms)+len(finished), :] = val        
        to_add[:, len(finished):len(ds_terms)+len(finished)] = val
        ###Subtract center square
        to_add[len(finished):len(ds_terms)+len(finished), len(finished):len(ds_terms)+len(finished)] = 0
        ###Add to matrix
        matrix_raw += to_add
        
    if len(putative_root.clades) == 2:
        matrix_weighted, matrix_raw, finished = pairwise_distances_recursive(putative_root.clades[0],\
                                                                      my_tree, matrix_weighted, matrix_raw,\
                                                                      term_map_dict, finished)
        matrix_weighted, matrix_raw, finished = pairwise_distances_recursive(putative_root.clades[1],\
                                                                      my_tree, matrix_weighted, matrix_raw,\
                                                                      term_map_dict, finished)
    elif len(putative_root.clades) == 0:
        finished.append(putative_root)
    return matrix_weighted, matrix_raw, finished





