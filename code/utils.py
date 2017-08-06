'''
Created on Aug 2, 2017

@author: alon
'''

def removeApostrophe(str):
    return str.replace("'", "")


def fileToDic(fileName):
    """
    take input file where each line is <key><separtor><value>
    and return a dictionary
    """
    dic = {}
    with open(fileName) as f:
        for line in f:
            #Parse line
            line = removeApostrophe(line)
            line = line.split('\t')
            assert len(line) == 5
            lid, question, canonical, logicalForm, isConsistent = line
            lid = int(lid)
            #Convert isConsistent to boolean
            assert isConsistent in ("True\n", "False\n"), "isConsistent: <" + str(isConsistent) + ">"
            isConsistent = isConsistent == "True\n"

            #Add line to dictionary
            if lid not in dic:
                dic[lid] = [question, [], []]
            #Add the logical form to the consistent or incosistent unique list
            if isConsistent:
                dic[lid][1] = list(set(dic[lid][1] + [canonical]))
            else:
                dic[lid][2] = list(set(dic[lid][2] + [canonical]))
    return dic
    
