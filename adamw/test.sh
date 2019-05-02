export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.00010
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.00020
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.00040
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.00060
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.00080
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.0010
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.0020
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.0040
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.0060
python3 david_net_decay.py --tpu_name=infer2 --warm_up=1 --learning_rate=0.0080
