import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)
from config.BasicConfig import opt
import json
from cv.ProcessImage import ProcessImage

def scan_img(root_p = os.path.join(root, 'data', opt.dataset_name)):
    res = []
    for r, dir, file in os.walk(root_p):
        for name in file:
            if name.endswith('jpg') or name.endswith('jpeg') or name.endswith('png'):
                res.append(os.path.join(r, name))
    return res

prim = ProcessImage()

def absolute_class(img_path):
    if img_path.find('melanoma') != -1:
        return 0
    else:
        return 1


def get_json(loc = os.path.join(root, opt.prior_knowledge_josn_path), call_back=absolute_class):
    res_dict = {}
    for i in scan_img():
        res_dict[i] = call_back(i)
    with open(loc, 'w') as f:
        json.dump(res_dict, f)
    print('Save File OK! ')




get_json()