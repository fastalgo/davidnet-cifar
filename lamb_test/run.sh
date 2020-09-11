#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
for i in 6
do
	for j in 0.02
	do
		echo "warmup $i, learning_rate $j"
		gsutil -m rm -R -f gs://cifar10-data/cifar10jobs/*
		python3 david_net.py --warm_up=$i --learning_rate=$j --weight_decay=1.0 --tpu_name=trump
	done
done
