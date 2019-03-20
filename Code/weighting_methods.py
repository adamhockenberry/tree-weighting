from collections import defaultdict
import numpy as np
import pandas as pd
import random

from collections import Counter

def normalize_GSC_weights(weights_dict, rooted_tree):
    '''
    Scratch function at the moment

    '''
    normed_weights_dict = {}
    depths_dict = rooted_tree.depths()
    for term in rooted_tree.get_terminals():
        normed_weights_dict[term] = [weights_dict[term][-1]/depths_dict[term]]
    return normed_weights_dict

def GSC_adhock_old(rooted_tree):
    '''
    This is a fairly straightforward and fast implementation of the GSC algorithm using a Bio.Phylo 
    tree object as input. The output dictionary has values that are lists where the final list element
    is the weight but it keeps track of the weight at each depth which is useful later.
    '''
    ###Initialize the weights dictionary with zeros
    weights_dict = {}
    for terminal in rooted_tree.get_terminals():
        weights_dict[terminal] = [0.0]

    ###This can probably be cleaned up to just specify that the root shouldn't have any branch length
    ###because I can't figure out why on earth it ever would
    ###This code just gets the depths of each terminal
    if rooted_tree.root.branch_length:
        bl = rooted_tree.root.branch_length
        all_depths_dict = rooted_tree.depths(unit_branch_lengths=True)
        for key,val in all_depths_dict.items():
            all_depths_dict[key] = int(val-bl)
    else:
        all_depths_dict = rooted_tree.depths(unit_branch_lengths=True)
    ###Little trick here to alter the all_depths_dict but unclear whether it does anything at the moment. WTF
    #all_depths_dict = {key: all_depths_dict[key] for key in rooted_tree.get_nonterminals()+rooted_tree.get_terminals()}
    all_depths_reverse_dict = defaultdict(list)
    for key, val in all_depths_dict.items():
        all_depths_reverse_dict[val].append(key)
    for i in range(max(all_depths_reverse_dict.keys()), 0, -1):
        for key,val in weights_dict.items():
            weights_dict[key].append(val[-1])
        for clade in all_depths_reverse_dict[i]:
            try:
                aterms = clade.clades[0].get_terminals()
                bterms = clade.clades[1].get_terminals()
            except IndexError:
                weights_dict[clade][-1] += clade.branch_length
                continue
            a = [weights_dict[term][-1] for term in aterms]
            b = [weights_dict[term][-1] for term in bterms]
            atot = np.sum(a)
            btot = np.sum(b)
            tot_weights = atot+btot
            #I think this is a buggy implementation that produces the correct results for an example
            #but makes less sense the more I think about it. The weight at each branch shouldn't be divided
            #equally among all sequences in the left sub-tree for instances and should rather be divided
            #to each sequence according to its weight
            if tot_weights != 0:
                afrac = (atot/tot_weights)/len(a)
                bfrac = (btot/tot_weights)/len(b)
            else:
                afrac = 0.5/len(a)
                bfrac = 0.5/len(b)
            for term in aterms:
                weights_dict[term][-1] += clade.branch_length*afrac
            for term in bterms:
                weights_dict[term][-1] += clade.branch_length*bfrac    
    return weights_dict

def GSC_adhock(rooted_tree):
    '''
    This modification just changes the way weights are partitioned. But I think it might have problems with leaves of
    zero branch length so might not make as much sense as I initially thought
    '''
    ###Initialize the weights dictionary with zeros
    weights_dict = {}
    for terminal in rooted_tree.get_terminals():
        weights_dict[terminal] = [0.0]

    ###This can probably be cleaned up to just specify that the root shouldn't have any branch length
    ###because I can't figure out why on earth it ever would
    ###This code just gets the depths of each terminal
    if rooted_tree.root.branch_length:
        bl = rooted_tree.root.branch_length
        all_depths_dict = rooted_tree.depths(unit_branch_lengths=True)
        for key,val in all_depths_dict.items():
            all_depths_dict[key] = int(val-bl)
    else:
        all_depths_dict = rooted_tree.depths(unit_branch_lengths=True)
    all_depths_reverse_dict = defaultdict(list)
    for key, val in all_depths_dict.items():
        all_depths_reverse_dict[val].append(key)
    for i in range(max(all_depths_reverse_dict.keys()), 0, -1):
        for key,val in weights_dict.items():
            weights_dict[key].append(val[-1])
        for clade in all_depths_reverse_dict[i]:
            try:
                aterms = clade.clades[0].get_terminals()
                bterms = clade.clades[1].get_terminals()
            except IndexError:
                weights_dict[clade][-1] += clade.branch_length
                continue
            a = [weights_dict[term][-1] for term in aterms]
            b = [weights_dict[term][-1] for term in bterms]
            tot_weights = np.sum(a+b)
            
            if tot_weights != 0:
                for term in aterms + bterms:
                    weights_dict[term][-1] += ((weights_dict[term][-1]/tot_weights)*clade.branch_length)
            else:
                for term in aterms + bterms:
                    weights_dict[term][-1] += clade.branch_length/len(aterms+bterms)
    return weights_dict


###################
#ACL weights#######
###################

###NOTE: I'd like to think of a way to normalize these weights better
def ACL_adhock(rooted_tree):
    assert rooted_tree.is_bifurcating()
    zero_bl_clades = [term for term in rooted_tree.get_terminals() if term.branch_length==0.]
    if len(zero_bl_clades) != 0:
        rooted_tree = trim_zero_bls(rooted_tree)
        print('Found some zero branch length terminals and have removed them. Note that '
                'returned tree will contain fewer taxa than original')
    if rooted_tree.root.branch_length == None:
        rooted_tree.root.branch_length = 0.
    initial_order = rooted_tree.get_terminals()
    initial_matrix = np.zeros((len(initial_order),len(initial_order)))
    ###Calling the recursive function
    vcv_matrix, finished_list = vcv_recursive(rooted_tree.root, initial_matrix, [])
    ###And cleaning things up
    inv_vcv_matrix = np.linalg.inv(vcv_matrix)
    inv_weights = inv_vcv_matrix.sum(axis=1)/inv_vcv_matrix.sum()
    rooted_tree.root.branch_length = None
    weights_dict = dict(zip(initial_order, inv_weights))
    return weights_dict, rooted_tree

def vcv_recursive(putative_root, vcv_matrix, finished):
    '''
    Requires a rooted tree (binary root) and that root length should be zero. Also believe some zero bl's will royally screw things up
    during matrix inversion
    '''
    terminals = putative_root.get_terminals()
    if not set(terminals).issubset(set(finished)):
        vcv_matrix[len(finished):len(finished)+len(terminals), len(finished):len(finished)+len(terminals)] += putative_root.branch_length
    if len(putative_root.clades) == 2:
            vcv_matrix, finished = vcv_recursive(putative_root.clades[0], vcv_matrix, finished)
            vcv_matrix, finished = vcv_recursive(putative_root.clades[1], vcv_matrix, finished)
    elif len(putative_root.clades) == 0:
        finished.append(putative_root)
    else:
        print("ERROR: APPEARS TO BE A NON-BINARY TREE. MATRIX GENERATION WILL PROBABLY FAIL")
    return vcv_matrix, finished




#def get_vcv_recursive(vcv_matrix, initial_clade, finished=[]):
#   '''This is an old function that shoudl probably be ignored for now'''
#    if len(initial_clade) == 2:
#        if not set(initial_clade[0].get_terminals()).issubset(set(finished)):
#            clade = initial_clade[0]
#            vcv_matrix[len(finished):len(finished)+len(clade.get_terminals()), len(finished):len(finished)+len(clade.get_terminals())] += clade.branch_length
#            vcv_matrix, finished = get_vcv_recursive(vcv_matrix, clade, finished)
#
#        if not set(initial_clade[1].get_terminals()).issubset(set(finished)):
#            clade = initial_clade[1]
#            vcv_matrix[len(finished):len(finished)+len(clade.get_terminals()), len(finished):len(finished)+len(clade.get_terminals())] += clade.branch_length
#            vcv_matrix, finished = get_vcv_recursive(vcv_matrix, clade, finished)
#    elif len(initial_clade) == 0:
#        finished.append(initial_clade)
#
#    else:
#        print("ERROR: APPEARS TO BE A NON-BINARY TREE. MATRIX GENERATION WILL PROBABLY FAIL")
#    return vcv_matrix, finished



#############Henikoff and Henikoff weights
def HH_adhock(records_list):
    """
    This is a modified Henikoff and Henikoff algorithm that I developed to correct
    for gapped sequences. The entire thing proceeds as usual but instead of either
    ignoring gaps or treating them as a 21st character, I give all gaps a weight of
    zero and furthermore I downweight each column of the alignment according to whatever
    it's weight *would* be times the number of ungapped positions/all positions.

    Essentially, if a column has a ton of gaps those gaps all get zero and also the
    remaining characters get heavily downweighted as well.

    Depending on the application, the normalization at the bottom is pretty critical and
    should be considered (divide all values by the max?, the mean?, etc.)
    """
    seqs = np.array([list(record.seq) for record in records_list])
    ids = [record.id for record in records_list]
    seqs_T = seqs.T

    weights_T = []
    all_weights = []
    for i in seqs_T[:]:
        counter_dict = Counter(i)
        ###Comment/delete the below line to treat gaps as a 21st character
        del counter_dict['-']
        
        r = len(counter_dict.keys())
        weights_dict = {}
        for key, val in counter_dict.items():
            weights_dict[key] = 1./(r*val)
        
        ###Note that implicitly here gapped sequences will get zero weight if gaps
        ###were deleted from the counter_dict above
        temp_array = np.zeros(i.shape)
        for key, val in weights_dict.items():
            np.place(temp_array, i==key, [val])
        #Rescale to punish gapped columns    
        positions = np.sum(list(counter_dict.values())) #number of non-gapped positions
        temp_array = temp_array * (positions/seqs_T.shape[1])
        ###Append
        weights_T.append(temp_array)
    weights_T = np.array(weights_T)
    all_weights = weights_T.T
    all_weights = np.sum(all_weights, axis=1)
    all_weights = all_weights/np.max(all_weights)
    final_weights_dict = dict(zip(ids, all_weights))
    return final_weights_dict


def trim_zero_bls(my_tree):
    zero_bls = [term for term in my_tree.get_terminals() if term.branch_length==0.]
    while len(zero_bls)>0:
        my_tree.prune(random.choice(zero_bls))
        zero_bls = [term for term in my_tree.get_terminals() if term.branch_length==0.]
    return my_tree
