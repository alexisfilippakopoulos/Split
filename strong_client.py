from client import ClientTemplate 
import pickle
import socket
import sys
import threading
import time
from database import Database

BYTE_CHUNK = 4096

class StrongClient(ClientTemplate):
    def __init__(self, ip, client_port, server_port, ip_to_conn, port_to_conn, db: Database) -> None:
        super().__init__()
        print(self.device)
        # dict: weak_client_socket -> weak_client_address. Client address is used as a key in other dicts.
        self.my_ip = ip
        self.client_port = client_port
        self.ip_to_conn = ip_to_conn
        self.port_to_conn = port_to_conn
        self.clients_id_to_sock = []
        self.server_port = server_port
        self.db = db

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
                # Thread for establishing communication with the connected client
                threading.Thread(target=self.listen_for_server_sock_messages, args=(weak_client_socket, )).start()
                self.handle_connections_from_serversocket(weak_client_socket, weak_client_address)
        except socket.error as err:
            print(err)
            sys.exit(1)

    def handle_connections_from_serversocket(self, weak_client_socket, weak_client_address):
        client_ip, client_port = weak_client_address
        query = """
        SELECT id
        FROM clients
        WHERE ip = ? AND port = ?
        """
        exists = self.db.execute_query(query=query, values=(client_ip, client_port), fetch_data_flag=True, fetch_all_flag=True)
        if len(exists) == 0:
            query = """
            SELECT id FROM clients ORDER BY id DESC LIMIT 1;
            """
            last_id = self.db.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
            client_id = 1 if len(last_id) == 0 else last_id[0][0] + 1
            query = """
            INSERT INTO clients (id, ip, port) VALUES (?, ?, ?)
            """
            self.db.execute_query(query=query, values=(client_id, client_ip, client_port))
        else:
            client_id = exists[0][0]
        self.clients_id_to_sock[client_id] = weak_client_socket
        

    def listen_for_server_sock_messages(self, weak_client_socket, weak_client_address):
        data_packet = b''
        while True:
            data_chunk = weak_client_socket.recv(BYTE_CHUNK)
            data_packet += data_chunk
            if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        threading.Thread(target=self.handle_server_sock_packet, args=(data_packet, weak_client_socket)).start()
                        data_packet = b'' 
            if not data_chunk:
                break

    
    def handle_server_sock_packet(self, data_packet, weak_client_socket):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        print(data)
        self.send_data_packet('hi weak client', weak_client_socket)
        """for header, payload in data:
            # implement different functionality based on headers
            pass"""

    def handle_client_sock_packet(self, data_packet):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        print(data)
        """for header, payload in data:
            # implement different functionality based on headers
            pass"""

    def train_one_epoch():
        raise NotImplementedError("Subclasses should implement this method")
        # Implement different functionality


if __name__ == '__main__':
    table_queries = {
        'weak_clients' : """
                        CREATE TABLE weak_clients(
                        id INT PRIMARY KEY,
                        ip VARCHAR(50),
                        port INT,
                        datasize INT,
                        device VARCHAR(50))
                        """,
    }
    db = Database(db_path='strong_client.db', table_queries=table_queries)
    strong_client = StrongClient(ip='localhost', client_port=9999, server_port=6969, ip_to_conn='localhost', port_to_conn=10000, db=db)
    # Create socket that accepts connections from other clients and establishes communication
    strong_client.create_server_socket()
    # Thread for accepting incoming connections from clients
    threading.Thread(target=strong_client.listen_for_connections, args=()).start()
    # Create socket that connects with the server
    strong_client.create_client_socket(client_ip=strong_client.my_ip, client_port=strong_client.client_port, server_ip=strong_client.ip_to_conn, server_port=strong_client.port_to_conn)
    #Thread for establishing communication with the server
    threading.Thread(target=strong_client.listen_for_client_sock_messages, args=()).start()
    strong_client.send_data_packet('hiiii server', strong_client.client_socket)
