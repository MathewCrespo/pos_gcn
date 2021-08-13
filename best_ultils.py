import os
import numpy as np
import json

def parse_best(name):
    metric_record = {
        'Test/AUC':[],
        'Test/Acc':[],
        'Test/Malignant_precision':[],
        'Test/Malignant_recall':[],
        'Test/Benign_recall':[],
        'Test/Malignant_F1':[]
    }
    log_path = '/media/hhy/data/gcn_results/'+ name + '/log.txt'
    log = open(log_path,'r')
    lines = log.readlines()
    for line in lines:
        if line.startswith('Test/'):
            key = line.split(': ')[0]
            value = float(line.split(': ')[1].strip('\n'))
            if key in metric_record:
                metric_record[key].append(value)
    log.close()
    
    # to find the metrics with the highest auc
    auc = max(metric_record['Test/AUC'])
    idx = metric_record['Test/AUC'].index(auc)
    best_metric = {
        'epoch':idx,
        'auc':auc,
        'acc':metric_record['Test/Acc'][idx],
        'precision':metric_record['Test/Malignant_precision'][idx],
        'recall':metric_record['Test/Malignant_recall'][idx],
        'specificity':metric_record['Test/Benign_recall'][idx],
        'f1':metric_record['Test/Malignant_F1'][idx]

    }
    print(best_metric)
    save_path = '/media/hhy/data/gcn_results/'+ name + '/best.json'
    js = json.dumps(best_metric)
    fp = open(save_path,'a')
    fp.write(js)
    fp.close()


if __name__ == '__main__':
    parse_best('hx_inception_adm') 
    