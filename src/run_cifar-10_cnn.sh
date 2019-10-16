./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn cnn --data_dir ${TF_IMAGE_DATA}/cifar-10-inc
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn cnn --data_dir ${TF_IMAGE_DATA}/cifar-10-inc >> cifar-10-inc.txt

echo "\n" >> cifar-10-inc.txt
cat experiments/base_model/params.json >> cifar-10-inc.txt