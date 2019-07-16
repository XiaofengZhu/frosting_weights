import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0',
                    help="retrain loss_fn")
parser.add_argument('--loss_fn', default='cnn',
                    help="retrain loss_fn")
parser.add_argument('--use_kfac', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="usek fac true gradient")
parser.add_argument('--log', default='',
                    help="log file")
parser.add_argument('--data_dir', default='../data/mnist-aug',
                    help="Directory containing the dataset")
args = parser.parse_args()

model_dir = 'experiments/base_model'
train_log = os.path.join(model_dir, 'train_{}.log'.format(args.log))
test_log = os.path.join(model_dir, 'test_{}.log'.format(args.log))

train_script = 'CUDA_VISIBLE_DEVICES={} python main.py --data_dir ../data/{} \
 --loss_fn {} --finetune true --use_kfac {} --log {}'.format(args.gpu, args.data_dir, args.loss_fn, \
 	args.use_kfac, args.log)
os.system('echo {} >> {}'.format(train_script, train_log))
os.system(train_script)
test_fake_script = 'python evaluate.py --data_dir ../data/{} \
 --loss_fn {} --finetune true --use_kfac {} --log {}'.format(args.loss_fn, \
 	args.data_dir, args.use_kfac, args.log)
os.system('echo {} >> {}'.format(test_fake_script, test_log))
test_script = 'CUDA_VISIBLE_DEVICES={} python evaluate.py --data_dir ../data/{} \
 --loss_fn {} --finetune true --log {}'.format(args.gpu, args.data_dir, args.loss_fn, args.log)	
os.system(test_script)