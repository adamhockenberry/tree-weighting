import numpy as np

def GSC_ete3(my_tree, weight=0):
    """
    A super simple implementation using the ete3 data structure. 
    And it checks out as far as I'm concerned, but probably needs some work around the edges.
    
    One ideological flaw that I see with the method, however, is this:
    test_tree = ete3.Tree('(((A:20, B:0):30,C:50):30, D:80);') 
    Node "B" should get zero weight as if it's not there?
    
    For the implementation, I should have a very short wrapper that checks the tree structure 
    for errors and then checks the output for consistency/expectation. Also (probably) removes
    the weights for internal nodes since these are just place-holders and need to be removed.
    """
    my_tree.add_features(weight=weight)
    if len(my_tree.get_children()) == 0:
        return
    elif len(my_tree.get_children())==2:
        l_child = my_tree.children[0]
        r_child = my_tree.children[1]
        l_ds = np.sum([i.dist for i in l_child.traverse()])
        r_ds = np.sum([i.dist for i in r_child.traverse()])
        total = l_ds + r_ds
        if total != 0:
            l_push = my_tree.weight * (l_ds/total)
            r_push = my_tree.weight * (r_ds/total)
        else:
            l_push = my_tree.weight/2.
            r_push = my_tree.weight/2.
        my_tree = GSC_ete3(l_child, l_push+l_child.dist)
        my_tree = GSC_ete3(r_child, r_push+r_child.dist)
    else:
        print('Error, tree does not appear to be bifurcating')
        return
    return
