# Split

### Use python < 3.12

### install torch with cuda support

### Run create_subsets.py once you clone

### Run src/server.py

### Run 

python src/strong_client.py -ip=127.0.0.1 --clientport=9999 --serverport=6969 --ip2connect=127.0.0.1 --port2connect=10000 --device=cuda --datapath=subset_data/sub_01.pth --batchsize=32 --epochs=20 --learningrate=0.01 --num_clients=2


### Run src/weal_client.py

