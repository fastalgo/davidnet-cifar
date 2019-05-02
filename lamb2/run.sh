#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for i in 5
do
	for j in 0.018 0.018 0.018
	do
		echo "warmup $i, learning_rate $j"
		gsutil -m rm -R -f gs://bert-pretrain-data/cifar/cifar10jobs/*
		python3 david_net.py --warm_up=$i --learning_rate=$j --weight_decay=0.01 --tpu_name=infer2
	done
done
