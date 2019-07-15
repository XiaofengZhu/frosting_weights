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
                    help="retrain loss_fn")
args = parser.parse_args()

model_dir = 'experiments/base_model'
train_log = os.path.join(model_dir, 'train{}.log'.format(args.log))
test_log = os.path.join(model_dir, 'test{}.log'.format(args.log))
for i in range(6, 11):
	train_script = 'CUDA_VISIBLE_DEVICES={} python main.py --train_range [1-{}] \
	 --loss_fn {} --finetune true --use_kfac {}'.format(args.gpu, i, args.loss_fn, args.use_kfac)
	os.system('echo {} >> {}'.format(train_script, train_log))
	os.system(train_script)
	test_fake_script = 'python evaluate.py --train_range [1-{}] \
	 --loss_fn {} --finetune true --use_kfac {}'.format(i, args.loss_fn, args.use_kfac)
	os.system('echo {} >> {}'.format(test_fake_script, test_log))
	test_script = 'CUDA_VISIBLE_DEVICES={} python evaluate.py \
	 --loss_fn {} --finetune true'.format(args.gpu, i, args.loss_fn)	
	os.system(test_script)