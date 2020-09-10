#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for k in 6 
do
	for i in 10.0
	do
		for j in 1.0
		do
			echo "warmup $k learning_rate $i, weight_decay $j"
			gsutil -m rm -R -f gs://cifar10-data/cifar/cifar10jobs/*
			python3 lamb_davidnet_cifar10.py --warm_up=$k --learning_rate=$i --weight_decay=$j --tpu_name=infer2
		done
	done
done
