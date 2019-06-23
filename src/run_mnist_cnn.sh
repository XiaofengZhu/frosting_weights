./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn cnn --data_dir ${TF_IMAGE_DATA}/mnist
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn cnn --data_dir ${TF_IMAGE_DATA}/mnist >> mnist.txt

echo "\n" >> mnist.txt
cat experiments/base_model/params.json >> mnist.txt