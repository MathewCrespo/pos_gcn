import os

for i in range(5):
    print('python main.py --log_root gcn_alnm{} --test_fold {} --config_name gcn_alnm{}'.format(str(i),str(i),str(i)))
    os.system('python main.py --log_root gcn_alnm{} --test_fold {} --config_name gcn_alnm{}'.format(str(i),str(i),str(i)))