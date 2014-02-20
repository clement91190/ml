filters = {
    'global': range(37),
    'q1': range(3),
    'q2': range(3, 5),
    'q3': range(5, 7),
    'q4': range(7, 9),
    'q5': range(9, 13),
    'q6': range(13, 15),
    'q7': range(15, 18),
    'q8': range(18, 25),
    'q9': range(25, 28),
    'q10': range(28, 31),
    'q11': range(31, 37),
    }


def complete(tab, learn_type):
    """ complete the results with 0. """
    filt = filters[learn_type]
    res = []
    j = 0
    for i in range(37):
        if i in filt:
            res.append(tab[j])
            j += 1
        else:
            res.append(0.)
    return res


