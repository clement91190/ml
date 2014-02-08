# -*-coding:Utf-8 -*
from mongoengine import Document, StringField, ListField, FloatField, BooleanField, IntField
from mongoengine import ObjectIdField, connect

#OPTIM_URI = 'mongodb://clement91190:rafiki@lafleur.mongohq.com:10078/app18535327'
#OPTIM_ALIAS = 'question-tree-production'

print "... connecting to database"
connect('optim_db')
print "...done"


class OptimProblem(Document):
    description = StringField()


class ParamInstance(Document):
    params = ListField()
    score = FloatField()
    optim_run = IntField()  # optim_run is a assigned to a specific algorithm
    optim_problem = ObjectIdField()

    done = BooleanField()
    assigned = BooleanField()






