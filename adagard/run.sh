#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for i in 1 2 3 4 5 6 7 8
do
	for j in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.011 0.012 0.013 0.014 0.015 0.02 0.03 0.04 0.05
	do
		echo "warmup $i, learning_rate $j"
		gsutil -m rm -R -f gs://bert-pretrain-data/cifar/cifar10jobs/*
		python3 david_net.py --warm_up=$i --learning_rate=$j --weight_decay=0.01
	done
done
