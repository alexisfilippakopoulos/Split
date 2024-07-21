import socket
import threading
import pickle
import sys
import argparse
import torch
from database import Database
from server_model import ServerModel
import pandas as pd
import numpy as np


BYTE_CHUNK = 4096
EPOCH = 0

class Server():
    def __init__(self, my_ip, my_port, db) -> None:
        self.server_ip = my_ip
        self.server_port = my_port
        self.db = db
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.client_outputs = {}
        self.client_labels = {}
        torch.manual_seed(32)
        self.server_model = ServerModel()
        self.strong_clients = []
        self.ids_to_datasize = {}
        self.losses = {}
        self.avg_losses = {}
        self.df = pd.DataFrame(columns=['epoch', 'client_id', 'avg_train_loss'])


    def create_server_socket(self):
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.server_socket.bind((self.server_ip, self.server_port))
        print(f'[+] Server initialized successfully at {self.server_ip, self.server_port}')\

    def listen_for_connections(self):
        self.server_socket.listen()
        while True:
            client_socket, client_address = self.server_socket.accept()
            print(f"[+] Client {client_address} connected")
            self.strong_clients.append(client_socket)
            threading.Thread(target=self.listen_for_messages, args=(client_socket, client_address)).start()

    def handle_connections(self):
         pass
    # todo implement handle connections and  call it from listen for conns

    def listen_for_messages(self, client_socket, client_address):
        # Communication thread with each weak client
        self.send_data('hiii strong client', client_socket)
        data_packet = b''
        while True:
            data_chunk = client_socket.recv(BYTE_CHUNK)
            data_packet += data_chunk
            if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        threading.Thread(target=self.handle_data_packet, args=(data_packet, client_socket)).start()
                        data_packet = b'' 
            if not data_chunk:
                break

    def handle_data_packet(self, data_packet, client_socket):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        if data.__class__ == dict:
            headers = list(data.keys())
            if 'inputs' in headers:
                self.client_outputs = data['inputs']
                self.client_labels = data['labels']
                self.update_models()
            elif 'datasizes' in headers:
                self.ids_to_datasize = dict(zip(data['ids'], data['datasizes']))
                print(self.ids_to_datasize)
                self.federated_averaging()

            

    def send_data(self, data, comm_socket):
        comm_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')

    def update_models(self):
        criterion = torch.nn.CrossEntropyLoss()
        for cid in self.client_outputs.keys():
            # Load approprate weights
            query = 'SELECT model_weights FROM clients WHERE id = ?'
            weights = pickle.loads(self.db.execute_query(query, (cid,), fetch_data_flag=True))
            self.server_model.load_state_dict(weights)
            self.server_model.to(self.device)
            self.server_model.train()
            # Forward pass
            optimizer = torch.optim.SGD(params=self.server_model.parameters(), lr=0.01)
            optimizer.zero_grad()
            outputs = self.server_model(self.client_outputs[cid])
            # Backward pass
            loss = criterion(outputs, self.client_labels[cid])
            loss.backward()
            self.losses[cid] += loss.item()
            #print(f'Client {cid} Loss: {loss.item()}')
            # update
            optimizer.step()
            #print(f'Updated with client {cid} model instance')
            # store weights
            query = 'UPDATE clients SET model_weights = ? WHERE id = ?'
            self.db.execute_query(query=query, values=(pickle.dumps(self.server_model.state_dict()), cid))
        #print(losses)
        self.client_labels = {}
        self.client_outputs = {}
        self.send_data(data=b'<OK>', comm_socket=self.strong_clients[0])

    def federated_averaging(self):
        global EPOCH
        weights = []
        datasizes = []
        for cid, datasize in self.ids_to_datasize.items():
            query = 'SELECT model_weights FROM clients WHERE id = ?'
            weights.append(self.db.execute_query(query=query, values=(cid, ), fetch_data_flag=True))
            datasizes.append(datasize)
            avg_loss = self.losses[cid] / np.round(datasize / 32)
            self.df.loc[len(self.df)] = {'epoch': EPOCH, 'client_id': cid, 'avg_train_loss': avg_loss}
            self.df.to_csv('server_stats.csv')
            EPOCH += 1
            print(f'Client {cid}: Server-Side Average Training Loss: {avg_loss: .2f}')
        total_data = sum(datasizes)
        avg_weights = {}
        for i in range(len(weights)):
            weight_dict = pickle.loads(weights[i])
            for layer, params in weight_dict.items():
                if layer not in avg_weights.keys():
                    avg_weights[layer] = params * (datasizes[i] / total_data)
                else:
                    avg_weights[layer] += params * (datasizes[i] / total_data)

        for cid in self.ids_to_datasize.keys():
            query = 'UPDATE clients SET model_weights = ? WHERE id = ?'
            self.db.execute_query(query=query, values=(pickle.dumps(avg_weights), cid))
            self.losses[cid] = 0
        print('[+] Aggregated Global Model')
        self.send_data(data=b'<CONTINUE>', comm_socket=self.strong_clients[0])
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', default='127.0.0.1', help='ip of current machine', type=str)     
    parser.add_argument('-sp', '--serverport', help='port to accept connections from clients', type=int)  
    parser.add_argument('-d', '--device', default=None, help='available device to be used', type=str)  
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int)
    parser.add_argument('-lr', '--learningrate', help='learning rate', type=float)
    parser.add_argument('-numcl', '--num_clients', help='number of clients that will be served', type=int)   
    return parser    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    table_queries = {
        'clients' : """
                        CREATE TABLE clients(
                        id INT PRIMARY KEY,
                        datasize INT,
                        model_weights BLOB,
                        outputs BLOB,
                        labels BLOB)
                        """,
    }
    db = Database(db_path='server_db.db', table_queries=table_queries)
    server = Server(my_ip='localhost', my_port=10000, db=db)
    query = """
    INSERT INTO clients (id, datasize, model_weights, outputs, labels)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        model_weights = ?
    """
    db.execute_query(query=query, values=(-1, None, pickle.dumps(server.server_model.state_dict()), None, None, pickle.dumps(server.server_model.state_dict())))
    server.losses[-1] = 0
    for i in range(1, args.num_clients):
        server.losses[i] = 0
        db.execute_query(query=query, values=(i, None, pickle.dumps(server.server_model.state_dict()), None, None, pickle.dumps(server.server_model.state_dict())))
    server.create_server_socket()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    

