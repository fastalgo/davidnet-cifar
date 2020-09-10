#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for i in 6
do
	for j in 0.05 0.04 0.03 0.025 0.02 0.015  0.015 0.01
	do
		echo "warmup $i, learning_rate $j"
		gsutil -m rm -R -f gs://cifar10-data/cifar10jobs/*
		python3 david_net.py --warm_up=$i --learning_rate=$j --weight_decay=1.0 --tpu_name=trump
	done
done
