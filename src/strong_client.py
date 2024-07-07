from client import ClientTemplate 
from client_model import WeakClientOffloadedModel, StrongClientModel
import pickle
import socket
import sys
import threading
import time
from database import Database
import torch
import argparse

BYTE_CHUNK = 4096
OUTPUTS_LABELS_RECVD = threading.Event()
MODEL_LOCK = threading.Lock()

class StrongClient(ClientTemplate):
    def __init__(self, ip, client_port, server_port, ip_to_conn, port_to_conn, db: Database) -> None:
        super().__init__()
        # dict: weak_client_socket -> weak_client_address. Client address is used as a key in other dicts.
        self.my_ip = ip
        self.client_port = client_port
        self.ip_to_conn = ip_to_conn
        self.port_to_conn = port_to_conn
        self.clients_id_to_sock = {}
        self.server_port = server_port
        self.db = db
        torch.manual_seed(32)
        self.client_model = StrongClientModel()
        torch.manual_seed(32)
        self.offloaded_model = WeakClientOffloadedModel()

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
        update_db_headers = ['labels', 'device', 'outputs', 'datasize']
        headers = list(data.keys())
        for header in headers:
            payload = data[header]
            if header == 'device':
                payload = str(payload)

            if header in ['outputs', 'labels']:
                payload = pickle.dumps(payload)

            query = f'UPDATE weak_clients SET {header} = ? WHERE id = ?'
            self.db.execute_query(query=query, values=(payload, client_id))

        if 'outputs' in headers and 'labels' in headers:
            with MODEL_LOCK:
                threading.Thread(target=self.train_offloaded_models(client_id))


    def handle_client_sock_packet(self, data_packet):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        print(data)
        """for header, payload in data:
            # implement different functionality based on headers
            pass"""

    def train_my_model(self, epochs, lr, train_dl):
        optimizer = torch.optim.SGD(params=self.client_model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        self.client_model.to(self.device)
        for e in range(epochs):
            print(f'[+] Epoch {e + 1}')
            avg_loss = self.train_my_model_one_epoch(optimizer=optimizer, criterion=criterion, train_dl=train_dl)
            print(f'\tAverage Training Loss: {avg_loss :.2f}')

    def train_my_model_one_epoch(self, optimizer, criterion, train_dl):
        self.client_model.train()
        curr_loss = 0.
        for i, (inputs, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            inputs, labels = inputs.to(self.device) , labels.to(self.device)
            outputs = self.client_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
        return curr_loss / len(train_dl)
        

    def train_offloaded_models(self, client_id):
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
        
        optimizer = torch.optim.SGD(params=self.offloaded_model.parameters(), lr=0.01)
        optimizer.zero_grad()

        components['outputs'], components['labels'] = components['outputs'].to(self.device), components['labels'].to(self.device)

        components['outputs'].retain_grad() if components['device'] != self.device else None

        # Perform the forward and backward pass
        outputs = self.offloaded_model(components['outputs'])
        loss = criterion(outputs, components['labels'])
        loss.backward()
        #print(loss.item())
        optimizer.step()
        # Transmit offload layer's gradients back to client
        self.send_data_packet(payload={'grads': components['outputs'].grad.clone().detach().to(components['device']), 'loss': loss.item()}, comm_socket=self.clients_id_to_sock[client_id])
        #print('Updated and transmitted for client: ', client_id)
        query = "UPDATE weak_clients SET offloaded_weights = ? WHERE id = ?"
        self.db.execute_query(query=query, values=(pickle.dumps(self.offloaded_model.state_dict()), client_id))

            
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
                        offloaded_weights BLOB,
                        epoch_weights BLOB)
                        """,
    }
    db = Database(db_path='strong_client.db', table_queries=table_queries)
    strong_client = StrongClient(ip=args.ip, client_port=args.clientport, server_port=args.serverport, ip_to_conn=args.ip2connect, port_to_conn=args.port2connect, db=db)
    strong_client.device = torch.device(args.device) if args.device is not None else strong_client.device
    # Create socket that accepts connections from other clients and establishes communication
    strong_client.create_server_socket()
    # Thread for accepting incoming connections from clients
    threading.Thread(target=strong_client.listen_for_connections, args=()).start()
    # Create socket that connects with the server
    strong_client.create_client_socket(client_ip=strong_client.my_ip, client_port=strong_client.client_port, server_ip=strong_client.ip_to_conn, server_port=strong_client.port_to_conn)
    # Thread for establishing communication with the server
    threading.Thread(target=strong_client.listen_for_client_sock_messages, args=()).start()
    strong_client.send_data_packet('hiiii server', strong_client.client_socket)

    train_dl, datasize = strong_client.load_data(subset_path=args.datapath, batch_size=32, shuffle=True, num_workers=2)
    threading.Thread(target=strong_client.train_my_model, args=(args.epochs, args.learningrate, train_dl)).start()
    
