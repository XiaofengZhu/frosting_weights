import os

model_dir = 'experiments/base_model'
train_log = os.path.join(model_dir, 'train.log')
test_log = os.path.join(model_dir, 'test.log')
for i in range(6, 11):
	train_script = 'CUDA_VISIBLE_DEVICES=1 python main.py --train_range [1-{}] \
	 --loss_fn retrain_regu_mine --finetune true'.format(i)
	os.system('echo {} >> {}'.format(train_script, train_log))
	os.system(train_script)
	test_fake_script = 'python evaluate.py --train_range [1-{}] \
	 --loss_fn retrain_regu_mine --finetune true'.format(i)
	os.system('echo {} >> {}'.format(test_fake_script, test_log))
	test_script = 'CUDA_VISIBLE_DEVICES=1 python evaluate.py \
	 --loss_fn retrain_regu_mine --finetune true'.format(i)	
	os.system(test_script)