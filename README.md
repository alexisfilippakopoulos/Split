# Split

### Use python < 3.12

### install torch with cuda support
---
### Run create_subsets.py

```python create_subsets.py```

Inside main there are the following function calls:

train_data, test_data = get_data(transform=transform, cifar_flag=True)

Where if you set cifar_flag=True you get CIFAR10 data and if you set cifar_flag=False you get FashionMNIST data.


create_subset(data=train_data,class_to_indices=class_to_indices, classes=[0], path='subset_data/sub_0.pth')

define the list of classes you want each subset to contain.

---
### Run Server

```python src/server.py --ip=127.0.0.1 --serverport=10000 --device=cuda --epochs=20 --learningrate=0.01 --num_clients=3```

--num_clients must be set as (num_strong + num_weak)

---

### Run Strong Client

```python src/strong_client.py -ip=127.0.0.1 --clientport=9999 --serverport=6969 --ip2connect=127.0.0.1 --port2connect=10000 --device=cuda --datapath=subset_data/sub_0.pth --batchsize=32 --epochs=20 --learningrate=0.01 --num_clients=2 --fedavg=2```

--num_clients must be set as the number of weak clients

--ip2connect and port2connect are the server's ip and port.

--fedavg is the epoch frequency for perfomring model aggregation

---

### Run Weak Client

```python src/weak_client.py -ip=127.0.0.1 --clientport=9998 --ip2connect=127.0.0.1 --port2connect=6969 --device=cuda --datapath=subset_data/sub_1.pth --batchsize=32 --epochs=20 --learningrate=0.01 --fedavg=2```

--ip2connect and port2connect are the strong client's ip and port.

--fedavg is the epoch frequency for perfomring model aggregation

---

### Metrics

For each experiment we produce 2 csv files:
client_stats.csv and server_stats.csv.

These files contain the following columns
epoch | client_id | client-side / server-siode average loss.

Also the following directory is created:

models with models/client and model/server

Where the aggregated models, every time we aggregate, for each side is saved.

Remember to create a folder e.g. Exeriments
with a subfolder for each run e.g. Exp1 and save there the csv files and the models directory for each experiment. We will need those for plotting / evaluation. If you dont the script will overwrite these for each run you make.