from client_model import ClientModel
import torch
import socket
import pickle

BYTE_CHUNK = 4096

# Contains the common functionality between the strong and weak client, defines a template method for the variable parts.
class ClientTemplate():
    def __init__(self) -> None:
        self.client_model = ClientModel()
        self.event_dict = {}
        self.device = self.get_device()

    def create_client_socket(self, client_ip, client_port, server_ip, server_port):
        try:
            self.client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.client_socket.bind((client_ip, client_port))
            self.client_socket.connect((server_ip, server_port))
            print(f'[+] Initialized socket at {(client_ip, client_port)} and connected with ({self.ip_to_conn}, {self.port_to_conn})')
        except socket.error as error:
            print(f'Socket initialization failed with error:\n{error}')
            print(self.client_socket.close())

    def listen_for_client_sock_messages(self):
        data_packet = b''
        try:
            while True:
                data_chunk = self.client_socket.recv(BYTE_CHUNK)
                if not data_chunk:
                    break
                data_packet += data_chunk
                if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                            self.handle_client_sock_packet(data_packet)
                            data_packet = b''
        except socket.error as error:
            print(f'Error receiving data:\n{error}')

    def handle_client_sock_packet(self, data_packet):
        raise NotImplementedError("Subclasses should implement this method")
        # Implement different functionality

    def train_one_epoch():
        raise NotImplementedError("Subclasses should implement this method")
        # Implement different functionality

    def get_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def send_data_packet(self, payload, comm_socket):
        try:
            comm_socket.sendall(b'<START>' + pickle.dumps(payload) + b'<END>')
        except socket.error as error:
            print(f'Message sending failed with error:\n{error}')