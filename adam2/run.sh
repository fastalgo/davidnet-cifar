#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for k in 5 0
do
	for i in 0.0001 0.0002 0.0004 0.0006 0.0008 0.001 0.002 0.004 0.006 0.008 0.01 0.02 0.04 0.06 0.08 0.1
	do
		for j in 1.0
		do
			echo "warmup $k learning_rate $i"
			gsutil -m rm -R -f gs://bert-pretrain-data/cifar/cifar10jobs/*
			python3 adam_davidnet_cifar10.py --warm_up=$k --learning_rate=$i --weight_decay=$j --tpu_name=infer2
		done
	done
done
