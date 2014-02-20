import numpy as np


class QuestionTree():
    """ on Tree per question """
    def __init__(self, question_id=1):
        self.sons = []
        self.question_id = question_id
        self.id = 0  #id correspond to the case in the tab where the probability of this answer is stored

    def add_son(self, son=None):
        if son is None:
            son = QuestionTree()
        self.sons.append(son)

    def add_sons(self, sons):
        for s in sons:
            self.sons.append(s)

def galaxy_tree():
    q1 = QuestionTree(1)
    q2 = QuestionTree(2)
    q3 = QuestionTree(3)
    q4 = QuestionTree(4)
    q5 = QuestionTree(5)
    q6 = QuestionTree(6)
    q7 = QuestionTree(7)
    q8 = QuestionTree(8)
    q9 = QuestionTree(9)
    q10 = QuestionTree(10)
    q11 = QuestionTree(11)
    qend = QuestionTree("end")

    q1.add_sons([q7, q2, qend])
    q2.add_sons([q9, q3])
    q3.add_sons([q4, q4])
    q4.add_sons([q10, q5])
    q5.add_sons([q6, q6, q6, q6])
    q6.add_sons([q8, qend])
    q7.add_sons([q6, q6, q6])
    q8.add_sons([qend, qend, qend, qend, qend, qend, qend])
    q9.add_sons([q6, q6, q6])
    q10.add_sons([q11, q11, q11])
    q11.add_sons([q5, q5, q5, q5, q5, q5])

    questions = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11]
    answers = []
    temp = [q.sons for q in questions]
    for s in temp:
        for a in s:
            answers.append(a)
    id = 0
    for q in questions:
        q.id = id
        id += len(q.sons)

    assert(len(answers) == 37)
    return questions, answers
    

class Simplifier():
    def __init__(self):
        self.questions, self.answers = galaxy_tree()
        self.thresold = 0.01  
        self.real_order = [1, 2, 3, 4, 10, 11, 5, 9, 7, 6, 8]

    def logic(self, tab):
        """ modify the 37 lenght array to account for the inference """

        questions_answered = [False for q in self.questions]
        answer_useful = [False for q in self.answers]
        questions_answered[0] = True
        answer_useful[0] = True
        answer_useful[1] = True
        answer_useful[2] = True
        for id_q in self.real_order:
            for ind, s in enumerate(self.questions[id_q - 1].sons):
                i = self.questions[id_q - 1].id + ind
                val = tab[i]
                if val > self.thresold and answer_useful[i]:
                    q_id = self.answers[i].question_id
                    if q_id != "end":
                        q_id -= 1
                        questions_answered[q_id] = True
                        question = self.questions[q_id]
                        for ans_id in range(question.id, question.id + len(question.sons)):
                            answer_useful[ans_id] = True
        modified_tab = []            
        for bol, v in zip(answer_useful, tab):
            if bol:
                modified_tab.append(v)
            else:
                modified_tab.append(0.)
        return modified_tab
   
    def normalize(self, tab):
        ind = 0
        for q in self.questions[1:]:
            s = np.sum(tab[ind:q.id])
            if s > 0:
                tab[ind:q.id] /= np.sum(tab[ind:q.id])
            ind = q.id
        last = 37
        s = np.sum(tab[ind:last])
        if s > 0:
            tab[ind:last] /= np.sum(tab[ind:last])
        return tab
 
    def modify_array(self, tab):
        tab = self.logic(tab)
        print tab
        tab = self.normalize(tab)
        print tab
        raw_input()
        return tab
    
    def modify_batch(self, batch):
        for i in range(batch.shape[0]):
            assert(batch[i].shape[0] == 37)
            batch[i] = self.modify_array(batch[i])
        return batch


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

def main():
    simpl = Simplifier()
    tab = np.ones(37)
    tab[0] = 0.0
    print simpl.modify_array(tab)

if __name__ == "__main__":
    main()
