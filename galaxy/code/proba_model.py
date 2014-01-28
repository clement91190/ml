import numpy as np


def from_array_to_model(tab):
    """ convert the 37 probabiliy array to the model containing only the 26 parameters """
    t = list(tab)
    q = [range(11)]
    q[0] = []
    q[1] = t[:2]
    q[2] = [t[3]]
    q[3] = [t[5]]
    q[4] = [t[7]]
    q[5] = t[9:12]
    q[6] = [t[13]]
    q[7] = t[15:17]
    q[8] = t[18:24]
    q[9] = t[25:27]
    q[10] = t[28:30]
    q[11] = t[31:36]

    proba = reduce(lambda x, y: x + y, q)
    assert(len(proba) == 26)
    proba = np.array(proba)
    return proba


def proba_of_being_asked_a_question(tab):
    """ return the proba that a question has been asked given the probabilities"""
    #TODO FINISH THIS
    proba = from_array_to_model(tab)
    q = [0 for i in range(11)]
    q[0] = 1.0
    q[6] = proba[1]
    q[1] = proba[0]
    q[8] = q[1] * (1 - proba[2]) 
    q[2] = q[1] * proba[2]
    q[3] = q[2]
    q[4] = q[3] * (1 - proba[4])
    q[5] = q[1] + q[6]
    q[7] = q[5] * proba[8]
    q[9] = q[3] * proba[4]
    q[10] = q[9]

    return q


def importance(tab):
    """ show the importance of the question if near 0. """
    pass
