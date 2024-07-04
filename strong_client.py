from client import ClientTemplate 
import pickle
import socket
import sys
import threading

BYTE_CHUNK = 4096

class StrongClient(ClientTemplate):
    def __init__(self, ip, client_port, server_port, ip_to_conn, port_to_conn) -> None:
        super().__init__(ip, client_port, ip_to_conn, port_to_conn)
        print(self.device)
        # dict: weak_client_socket -> weak_client_address. Client address is used as a key in other dicts.
        self.connected_clients_adds = {}
        self.server_port = server_port

    def create_server_socket(self):
        # Create a socket that is used for accepting connections and receiving data from weak clients
        try:
            self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.server_socket.bind(self.my_ip, self.server_port)
        except socket.error as err:
            print(err)
            sys.exit(1)

    def listen_for_connections(self):
        # Listen for connections from weak clients 
        try:
            self.server_socket.listen()
            while True:
                weak_client_socket, weak_client_address = self.server_socket.accept()
                # Thread for establishing communication with the connected client
                threading.Thread(target=self.listen_for_server_sock_messages, args=(weak_client_socket, weak_client_address)).start()
                print(f"[+] Weak client {weak_client_address} connected")
        except socket.error as err:
            print(err)
            sys.exit(1)

    def listen_for_server_sock_messages(self, weak_client_socket, weak_client_address):
        # Communication thread with each weak client
        self.connected_clients_adds[weak_client_socket] = weak_client_address
        data_packet = b''
        while True:
            data_chunk = weak_client_socket.recv(BYTE_CHUNK)
            data_packet += data_chunk
            if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        threading.Thread(target=self.handle_server_sock_packet, args=(data_packet, client_id)).start()
                        data_packet = b'' 
            if not data_chunk:
                break

            # TODO implement handle server_sock and client_sock

    
    def handle_server_sock_packet(self, data_packet):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        for header, payload in data:
            # implement different functionality based on headers
            pass

    def handle_client_sock_packet(self, data_packet):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        for header, payload in data:
            # implement different functionality based on headers
            pass

    def train_one_epoch():
        raise NotImplementedError("Subclasses should implement this method")
        # Implement different functionality


if __name__ == '__main__':
    strong_client = StrongClient(ip='localhost', client_port=9999, server_port=6969, ip_to_conn='localhost', port_to_conn=6000)
    # Create socket that connects with the server
    strong_client.create_client_socket()
    # Create socket that accepts connections from other clients
    strong_client.create_server_socket()
    # Thread for accepting incoming connections from clients
    threading.Thread(target=strong_client.listen_for_connections, args=()).start()
    #Thread for establishing communication with the server
    threading.Thread(target=strong_client.listen_for_client_sock_messages, args=()).start()