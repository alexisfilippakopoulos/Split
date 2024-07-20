from client import ClientTemplate
from client_model import WeakClientOffloadedModel, StrongClientModel
import pickle
import socket
import sys
import threading
from copy import deepcopy
from database import Database
import torch
import argparse
torch.autograd.set_detect_anomaly(True)

BYTE_CHUNK = 4096
OUTPUTS_LABELS_RECVD = threading.Event()
MODEL_LOCK = threading.Lock()

class StrongClient(ClientTemplate):
    def __init__(self, ip, client_port, server_port, ip_to_conn, port_to_conn, db: Database) -> None:
        super().__init__()
        self.my_ip = ip
        self.client_port = client_port
        self.ip_to_conn = ip_to_conn
        self.port_to_conn = port_to_conn
        self.server_port = server_port
        self.db = db
        # Maps a client's id to its socket
        self.clients_id_to_sock = {}
        # Track clients who finished their epoch and transmitted their updated weights
        self.client_updated_weigts = []
        torch.manual_seed(32)
        self.client_model = StrongClientModel()
        torch.manual_seed(32)
        self.offloaded_model = WeakClientOffloadedModel()
        self.optimizer = torch.optim.SGD(self.client_model.parameters(), lr=0.01)
        self.client_losses = {}
        self.client_outputs = {}
        self.client_off_outputs = {}
        self.client_labels = {}

    def create_server_socket(self):
        # Create a socket that is used for accepting connections and receiving data from weak clients
        try:
            self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.server_socket.bind((self.my_ip, self.server_port))
            print(f'[+] Initialized server socket at {(self.my_ip, self.server_port)}')
        except socket.error as err:
            print(err)
            sys.exit(1)

    def listen_for_connections(self):
        # Listen for connections from weak clients 
        try:
            print("[+] Listening for incoming connections")
            self.server_socket.listen()
            while True:
                weak_client_socket, weak_client_address = self.server_socket.accept()
                print(f"[+] Weak client {weak_client_address} connected")
                client_id = self.handle_connections_from_serversocket(weak_client_socket, weak_client_address)
                # Thread for establishing communication with the connected client
                threading.Thread(target=self.listen_for_server_sock_messages, args=(client_id, )).start()
        except socket.error as err:
            print(err)
            sys.exit(1)

    def handle_connections_from_serversocket(self, weak_client_socket, weak_client_address):
        client_ip, client_port = weak_client_address
        query = """
        SELECT id
        FROM weak_clients
        WHERE ip = ? AND port = ?
        """
        exists = self.db.execute_query(query=query, values=(client_ip, client_port), fetch_data_flag=True, fetch_all_flag=True)
        if len(exists) == 0:
            query = """
            SELECT id FROM weak_clients ORDER BY id DESC LIMIT 1;
            """
            last_id = self.db.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
            client_id = 1 if len(last_id) == 0 else last_id[0][0] + 1
            query = """
            INSERT INTO weak_clients (id, ip, port) VALUES (?, ?, ?)
            """
            self.db.execute_query(query=query, values=(client_id, client_ip, client_port))
        else:
            client_id = exists[0][0]
        self.clients_id_to_sock[client_id] = weak_client_socket
        query = "UPDATE weak_clients SET offloaded_weights = ? WHERE id = ?"
        self.db.execute_query(query=query, values=(pickle.dumps(self.offloaded_model.state_dict()), client_id))
        return client_id
        

    def listen_for_server_sock_messages(self, client_id):
        data_packet = b''
        while True:
            data_chunk = self.clients_id_to_sock[client_id].recv(BYTE_CHUNK)
            data_packet += data_chunk
            if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        threading.Thread(target=self.handle_server_sock_packet, args=(data_packet, client_id)).start()
                        data_packet = b'' 
            if not data_chunk:
                break

    
    def handle_server_sock_packet(self, data_packet, client_id):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        headers = list(data.keys())
        for header in headers:
            payload = data[header]
            if header == 'device':
                payload = str(payload)

            if header in ['outputs', 'labels', 'epoch_weights']:
                payload = pickle.dumps(payload)
                if header == 'epoch_weights':
                    print(f'[+] Received end of epoch weights from client {client_id}')
                    self.client_updated_weigts.append(client_id)

            query = f'UPDATE weak_clients SET {header} = ? WHERE id = ?'
            self.db.execute_query(query=query, values=(payload, client_id))

        if 'outputs' in headers and 'labels' in headers:
            with MODEL_LOCK:
                threading.Thread(target=self.forward_pass_offloaded_models(client_id))


    def handle_client_sock_packet(self, data_packet):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        print(data)
        """for header, payload in data:
            # implement different functionality based on headers
            pass"""
        
    def construct_weights_dicts(self):
        for client_id in self.client_updated_weigts:
            # Fetch both dicts offloaded_weights and epoch_weights
            query = "SELECT offloaded_weights FROM weak_clients WHERE id = ?"
            offload_dict = pickle.loads(self.db.execute_query(query=query, values=(client_id, ), fetch_data_flag=True))
            query = "SELECT epoch_weights FROM weak_clients WHERE id = ?"
            weak_model_dict = pickle.loads(self.db.execute_query(query=query, values=(client_id, ), fetch_data_flag=True))
            for layer, weights in offload_dict.items():
                weak_model_dict[layer] = weights
            # Save the full dict on epoch_weights
            query = "UPDATE weak_clients SET epoch_weights = ? WHERE id = ?"
            self.db.execute_query(query=query, values=(pickle.dumps(weak_model_dict), client_id))
        print("[+] Constructed all weak clients' weight dict")

    def federated_averaging(self):
        weights = []
        datasizes = []
        for client_id in self.client_updated_weigts:
            query = "SELECT epoch_weights FROM weak_clients WHERE id = ?"
            weights.append(pickle.loads(self.db.execute_query(query=query, values=(client_id, ), fetch_data_flag=True)))

            query = "SELECT datasize FROM weak_clients WHERE id = ?"
            datasizes.append(self.db.execute_query(query=query, values=(client_id, ), fetch_data_flag=True))

        weights.append(deepcopy(self.client_model.state_dict()))
        datasizes.append(self.datasize)
        total_data = sum(datasizes)
        # Aggregate the global model
        avg_full_weights = {}
        for i in range(len(weights)):
            for layer, weight in weights[i].items():
                weight = weight.to(self.device)
                if layer not in avg_full_weights.keys():
                    avg_full_weights[layer] = weight * (datasizes[i] / total_data)
                else:
                    avg_full_weights[layer] += weight * (datasizes[i] / total_data)
        # Create a state dict with the layers of the weak client model
        avg_weak_weights = {}
        for layer in avg_full_weights.keys():
            if layer not in self.offloaded_model.state_dict().keys():
                avg_weak_weights[layer] = avg_full_weights[layer]

        for client_id in self.client_updated_weigts:
            self.send_data_packet(payload={'avg_model': avg_weak_weights}, comm_socket=self.clients_id_to_sock[client_id])
            print(f"[+] Transmitted average model to client {client_id}")

        self.client_model.load_state_dict(avg_full_weights)
        self.client_updated_weigts.clear()
        del weights

    def train_my_model(self, num_clients, epochs, lr, train_dl):
        criterion = torch.nn.CrossEntropyLoss()
        self.client_model.to(self.device)
        for e in range(epochs):
            print(f'[+] Epoch {e + 1}')
            my_loss = self.train_my_model_one_epoch(criterion=criterion, train_dl=train_dl, num_clients=num_clients)
            # wait for all client to transmit their updated end of epoch weights
            while len(self.client_updated_weigts) < num_clients:
                continue
            # Reconstruct weight dicts
            self.construct_weights_dicts()
            # Perform the aggregation
            self.federated_averaging()


    def train_my_model_one_epoch(self, criterion, train_dl, num_clients):
        self.client_model.train()
        for i, (inputs, labels) in enumerate(train_dl):
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, server_inputs = self.client_model(inputs)
            #print('req', outputs.requires_grad)
            loss = criterion(outputs, labels)
            #outputs.retain_grad()
            #self.client_losses.append(loss.detach().cpu().numpy())
            self.client_losses[-1] = loss
            self.client_off_outputs[-1] = outputs
            self.client_outputs[-1] = server_inputs
            self.client_labels[-1] = labels
            while len(self.client_losses) < (num_clients + 1):
                continue
            self.update_models()
            del outputs

    def update_models(self):
        avg_loss = torch.stack(list(self.client_losses.values())).mean()
        #print(self.client_losses[-1].item(), self.client_losses[1].item(), self.client_losses[2].item())
        print(self.client_losses[-1].item(), self.client_losses[1].item())
        #print(avg_loss.item())
        avg_loss.backward(retain_graph=True)
        for cid, outputs in self.client_off_outputs.items():
            if cid == -1:
                self.optimizer.step()
            else:
                new_outs = outputs.clone().detach().requires_grad_(True)
                query = 'SELECT offloaded_weights FROM weak_clients WHERE id = ?'
                weight_dict = pickle.loads(self.db.execute_query(query=query, values=(cid, ), fetch_data_flag=True))
                query = f'SELECT device FROM weak_clients WHERE id = ?'
                device = self.db.execute_query(query=query, values=(cid, ), fetch_data_flag=True)
                self.offloaded_model.load_state_dict(weight_dict)
                optimizer = torch.optim.SGD(self.offloaded_model.parameters(), lr=0.01)
                optimizer.zero_grad()
                new_outs.backward(outputs.grad.clone())
                optimizer.step()
                #print(outputs.grad.shape)
                outputs.retain_grad()
                new_outs.retain_grad()
                #print(self.client_outputs[cid].grad)
                print(f'Transmitted grads to client {cid}')
                self.send_data_packet(payload={'grads': self.client_outputs[cid].grad.clone().detach().to(device)}, comm_socket=self.clients_id_to_sock[cid])
                #self.send_data_packet(payload={'grads': outputs.grad.clone().detach().to(device)}, comm_socket=self.clients_id_to_sock[cid])

        self.client_losses.clear()
        print(f'Transmitted features to client server')
        self.send_data_packet(payload={'inputs': self.client_outputs, 'labels': self.client_labels}, comm_socket=self.client_socket)
        self.client_off_outputs = {}
        self.client_outputs = {}
        self.client_labels = {}
        #print('OUTPUTS', self.client_outputs)

    def forward_pass_offloaded_models(self, client_id):
        # Fetch client's offloaded weights, outputs, labels and store them in a list
        components = {}
        criterion = torch.nn.CrossEntropyLoss()
    
        for col in ['offloaded_weights', 'outputs', 'labels']:
            query = f'SELECT {col} FROM weak_clients WHERE id = ?'
            components[col] = (pickle.loads(self.db.execute_query(query=query, values=(client_id, ), fetch_data_flag=True)))
        
        query = f'SELECT device FROM weak_clients WHERE id = ?'
        components['device'] = self.db.execute_query(query=query, values=(client_id, ), fetch_data_flag=True)

        # Load specific client's offloaded weights
        #print(self.device)
        self.offloaded_model.load_state_dict(components['offloaded_weights'])
        self.offloaded_model.to(self.device)
        self.offloaded_model.train()

        components['outputs'], components['labels'] = components['outputs'].to(self.device), components['labels'].to(self.device)
        #components['outputs'] = components['outputs'].clone().detach().requires_grad_(True)
        #components['outputs'].retain_grad() if components['device'] != self.device else None
        # Perform the forward and backward pass
        outputs = self.offloaded_model(components['outputs'])
        #self.client_losses.append(loss.detach().cpu().numpy())
        #print(components['outputs'].requires_grad_())
        components['outputs'].retain_grad()
        outputs.retain_grad()
        #outputs = outputs.clone().detach().requires_grad_(True)
        #self.client_outputs[client_id] = components['outputs'].clone().detach().requires_grad_(True)
        self.client_outputs[client_id] = components['outputs']
        self.client_off_outputs[client_id] = outputs
        loss = criterion(outputs, components['labels'])
        self.client_losses[client_id] = loss
        self.client_labels[client_id] = components['labels']

            
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', default='127.0.0.1', help='ip of current machine', type=str)
    parser.add_argument('-cp', '--clientport', help='port to connect with the server', type=int)       
    parser.add_argument('-sp', '--serverport', help='port to accept connections from clients', type=int)  
    parser.add_argument('-ip2con', '--ip2connect', default='127.0.0.1', help='ip of server', type=str)
    parser.add_argument('-p2con', '--port2connect', help='server port that accepts connections', type=int) 
    parser.add_argument('-d', '--device', default=None, help='available device to be used', type=str)  
    parser.add_argument('-data', '--datapath', help='path to a data subset', type=str)
    parser.add_argument('-bs', '--batchsize', help='size of batch', type=int) 
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int)
    parser.add_argument('-lr', '--learningrate', help='learning rate', type=float)
    parser.add_argument('-numcl', '--num_clients', help='number of clients that will be served', type=int)   
    return parser    


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    table_queries = {
        'weak_clients' : """
                        CREATE TABLE weak_clients(
                        id INT PRIMARY KEY,
                        ip VARCHAR(50),
                        port INT,
                        datasize INT,
                        device VARCHAR(50),
                        outputs BLOB,
                        labels BLOB,
                        loss BLOB,
                        offloaded_weights BLOB,
                        epoch_weights BLOB)
                        """,
    }
    db = Database(db_path='strong_client.db', table_queries=table_queries)
    strong_client = StrongClient(ip=args.ip, client_port=args.clientport, server_port=args.serverport, ip_to_conn=args.ip2connect, port_to_conn=args.port2connect, db=db)
    strong_client.device = torch.device(args.device) if args.device is not None else strong_client.device
    print(f'[+] Using {strong_client.device} as device.')
    # Create socket that accepts connections from other clients and establishes communication
    strong_client.create_server_socket()
    # Thread for accepting incoming connections from clients
    threading.Thread(target=strong_client.listen_for_connections, args=()).start()
    # Create socket that connects with the server
    strong_client.create_client_socket(client_ip=strong_client.my_ip, client_port=strong_client.client_port, server_ip=strong_client.ip_to_conn, server_port=strong_client.port_to_conn)
    # Thread for establishing communication with the server
    threading.Thread(target=strong_client.listen_for_client_sock_messages, args=()).start()
    #strong_client.send_data_packet('hiiii server', strong_client.client_socket)
    train_dl = strong_client.load_data(subset_path=args.datapath, batch_size=32, shuffle=True, num_workers=2)
    threading.Thread(target=strong_client.train_my_model, args=(args.num_clients, args.epochs, args.learningrate, train_dl)).start()
    