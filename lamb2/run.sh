#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for i in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5
do
	for j in 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0
	do
		echo "weight_decay $i, learning_rate $j"
		gsutil -m rm -R -f gs://bert-pretrain-data/cifar/cifar10jobs/*
		python3 david_net.py --warm_up=5 --learning_rate=$j --weight_decay=$i --tpu_name=infer2
	done
done
