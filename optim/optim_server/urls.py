# -*-coding:Utf-8 -*
from flask import request
from optim_server import app
from optim_server.db import models

import json


# Worker functions
@app.route('/get_params')
def get_params():
    try:
        obj = models.ParamInstance.objects(assigned=False, done=False).order_by('id')
        obj = obj[0]
        obj.assigned = True
        obj.save()
        return obj.to_json()
    except:
        return json.dumps(None)


@app.route('/post_results', methods=['GET'])
def post_results():
    res = request.args.get('res')
    id = request.args.get('id')
    res = float(res)
    obj = models.ParamInstance.objects.get(id=id)
    obj.done = True
    obj.assigned = False
    obj.score = res
    obj.save()
    return json.dumps(None)


#TaskManager urls
@app.route('/post_params', methods=['POST'])
def post_params():
    req_json = request.get_json()
    print req_json
    data = req_json
    params = data['params']
    print params
    try:
        optim_run = int(data['optim_run'])
    except:
        optim_run = 0
    params = [float(p) for p in params]
    obj = models.ParamInstance(params=params, done=False, assigned=False, optim_run=optim_run)
    obj.save()
    return str(obj.id)


@app.route('/get_results', methods=['GET'])
def get_results():
    """ return the result -> return None if no results. """
    id = request.args.get('id')
    obj = models.ParamInstance.objects.get(id=id)
    #return obj.to_json()
    return json.dumps({'res': obj.score})
