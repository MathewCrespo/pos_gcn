import os
import numpy as np
import json

def parse_best(name,mode='overall'):
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

    if mode == 'overall':
        n = np.array([metric_record['Test/AUC'],metric_record['Test/Acc'],metric_record['Test/Malignant_precision'],
                        metric_record['Test/Malignant_recall'],metric_record['Test/Benign_recall'],metric_record['Test/Malignant_F1']])
        overall = np.sum(n,axis=0)
        idx = np.argmax(overall)
    
    if mode == 'auc':
    # to find the metrics with the highest auc

        auc = max(metric_record['Test/AUC'])
        idx = metric_record['Test/AUC'].index(auc)
    best_metric = {
        'epoch':str(idx),
        'auc':metric_record['Test/AUC'][idx],
        'acc':metric_record['Test/Acc'][idx],
        'precision':metric_record['Test/Malignant_precision'][idx],
        'recall':metric_record['Test/Malignant_recall'][idx],
        'specificity':metric_record['Test/Benign_recall'][idx],
        'f1':metric_record['Test/Malignant_F1'][idx]

    }
    print(best_metric)
    save_path = '/media/hhy/data/gcn_results/'+ name + '/'+ mode +'.json'
    js = json.dumps(best_metric)
    fp = open(save_path,'a')
    fp.write(js)
    fp.close()

if __name__ == '__main__':
    for i in range(4):
        parse_best('hxALNM_{}'.format(i+1)) 
    