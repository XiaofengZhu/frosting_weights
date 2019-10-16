./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn cnn --data_dir ../mnist-inc
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn cnn --data_dir ../mnist-inc >> mnist-inc.txt

echo "\n" >> mnist-inc.txt
cat experiments/base_model/params.json >> mnist-inc.txt