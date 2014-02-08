# -*-coding:Utf-8 -*
import urllib2
import json
import time


def eval_fitness_function(tab, optim_run=0):
    """ tab is a list of parameters """
    pass


def get_results(id):
    url = "http://127.0.0.1:5000/get_results?id={}".format(id)
    req = urllib2.Request(url)
    res = None
    while res is None:
        f = urllib2.urlopen(req)
        res = f.read()
        res = json.loads(res)
        res = res['res']
        time.sleep(0.1)
    return res


def send_parameters(params, optim_run):
    data = {'params': range(5), 'optim_run': 0}
    data = json.dumps(data)
    url = "http://127.0.0.1:5000/post_params"
    req = urllib2.Request(url, data, {"Content-Type": 'application/json'})
    f = urllib2.urlopen(req)
    res = f.read()
    return res
