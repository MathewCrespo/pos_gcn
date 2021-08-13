import os
'''
for i in range(4):
    i += 1
    print('python main.py --log_root baseBM_res10_{} --test_fold {} --net Res_Attention'.format(str(i),str(i)))
    os.system('python main.py --log_root baseBM_res10_{} --test_fold {} --net Res_Attention'.format(str(i),str(i)))

for i in range(5):
    print('python main.py --log_root MyBM_res10_{} --test_fold {} --net H_Attention_Graph'.format(str(i),str(i)))
    os.system('python main.py --log_root MyBM_res10_{} --test_fold {} --net H_Attention_Graph'.format(str(i),str(i)))

for i in range(5):
    print('python main.py --log_root HX_ALNM{} --test_fold {} --net HX_Attention --lr 1e-4'.format(str(i),str(i)))
    os.system('python main.py --log_root HX_ALNM{} --test_fold {} --net HX_Attention --lr 1e-4'.format(str(i),str(i)))



for i in range(5):
    print('python main.py --log_root myALNM10_{} --test_fold {} --net H_Attention_GraphV3'.format(str(i),str(i)))
    os.system('python main.py --log_root myALNM10_{} --test_fold {} --net H_Attention_GraphV3'.format(str(i),str(i)))

for i in range(5):
    print('python main.py --log_root myALNM34_{} --test_fold {}'.format(str(i),str(i)))
    os.system('python main.py --log_root myALNM34_{} --test_fold {}'.format(str(i),str(i)))
'''
for i in range(5):
    for j in [0.1,0.2]:
        os.system('python main.py --log_root alnmt0.1_{} --test_fold {} --t {} --net H_Attention_GraphV3'.format(str(i),str(i),j))

