import socket
import threading
import pickle
import argparse
import torch
from database import Database
from server_model import ServerModel
from copy import deepcopy


BYTE_CHUNK = 4096
MODEL_LOCK = threading.Event()

class Server():
    def __init__(self, my_ip, my_port, db, num_clients) -> None:
        self.server_ip = my_ip
        self.server_port = my_port
        self.db = db
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.manual_seed(32)
        self.model = ServerModel()
        query = "INSERT INTO clients (id, datasize, device, server_model_weights, batches) VALUES (?, ?, ?, ?, ?)"
        for i in range(num_clients):
            self.db.execute_query(query=query, values=(i, None, None, pickle.dumps(self.model.state_dict()), pickle.dumps([])))
        self.client_ids = []
        self.clients_in_queue = set()
        self.current_id = -1
        self.finished_clients = []

    def create_server_socket(self):
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.server_socket.bind((self.server_ip, self.server_port))
        print(f'[+] Server initialized successfully at {self.server_ip, self.server_port}')

    def listen_for_connections(self):
        self.server_socket.listen()
        while True:
            client_socket, client_address = self.server_socket.accept()
            print(f"[+] Client {client_address} connected")
            threading.Thread(target=self.listen_for_messages, args=(client_socket, )).start()

    def handle_connections(self):
         pass
    # todo implement handle connections and  call it from listen for conns

    def listen_for_messages(self, client_socket):
        # Communication thread with each weak client
        data_packet = b''
        while True:
            data_chunk = client_socket.recv(BYTE_CHUNK)
            data_packet += data_chunk
            if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        threading.Thread(target=self.handle_data_packet, args=(data_packet, )).start()
                        data_packet = b'' 
            if not data_chunk:
                break

    def handle_data_packet(self, data_packet):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        headers = list(data.keys()) 
        if 'batch' in headers:
            client_id = data['id']
            # We have a packet of {'id': client_id, 'batch' : [outputs, labels]]}
            while client_id == self.current_id:
                pass
            # Paizei na thelei lock
            print('storing batches for client', client_id)
            query = "SELECT batches FROM clients WHERE id = ?"
            batches = pickle.loads(self.db.execute_query(query=query, values=(client_id,), fetch_data_flag=True))
            batches.append(data['batch'])
            query = "UPDATE clients SET batches = ? WHERE id = ?"
            self.db.execute_query(query=query, values=(pickle.dumps(batches), client_id))
            self.clients_in_queue.add(client_id)
        elif 'final' in headers:
            self.finished_clients.append(data['final'])

            
    def send_data(self, data, comm_socket):
        comm_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')

    def train_some_batches(self, batches):
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.01)
        curr_loss = 0
        for (inputs, labels) in batches:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
        return curr_loss / len(batches)

    def train_one_epoch(self, num_clients):
        while (len(self.clients_in_queue) == 0) or (len(self.finished_clients) < num_clients):
            self.current_id = -1
            while (len(self.clients_in_queue) == 0):
                pass
            print(self.current_id)
            self.current_id = self.clients_in_queue.pop()
            print(f"[+] Training with client {self.current_id}")
            query = 'SELECT server_model_weights FROM clients WHERE id = ?'
            self.model.load_state_dict(pickle.loads(self.db.execute_query(query=query, values=(self.current_id, ), fetch_data_flag=True)))
            query = "SELECT batches FROM clients WHERE id = ?"
            # batches = [[outputs1, labels1], ..., [outputsN, labelsN], ]
            batches = pickle.loads(self.db.execute_query(query=query, values=(self.current_id,), fetch_data_flag=True))
            avg_loss = self.train_some_batches(batches=batches)
            print(f"[+] Average Loss with client {self.current_id} for {len(batches)} batches: {avg_loss}")
            batches.clear()
            query = "UPDATE clients SET server_model_weights = ?, batches = ?"
            self.db.execute_query(query=query, values=(pickle.dumps(self.model.state_dict()), pickle.dumps(batches)))





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
                        device VARCHAR(50),
                        server_model_weights BLOB,
                        batches BLOB)
                        """,
    }
    db = Database(db_path='server_db.db', table_queries=table_queries)
    server = Server(my_ip=args.ip, my_port=args.serverport, db=db, num_clients=args.num_clients)
    server.create_server_socket()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    threading.Thread(target=server.train_one_epoch, args=(args.num_clients, )).start()
    

