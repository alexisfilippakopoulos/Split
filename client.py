from client_model import ClientModel
import torch
import socket
import pickle

BYTE_CHUNK = 4096

# Contains the common functionality between the strong and weak client, defines a template method for the variable parts.
class ClientTemplate():
    def __init__(self, my_ip, my_port, ip_to_conn, port_to_conn) -> None:
        self.my_ip = my_ip
        self.my_port = my_port
        self.ip_to_conn = ip_to_conn
        self.port_to_conn = port_to_conn
        self.client_model = ClientModel()
        self.event_dict = {}
        self.device = self.get_device()

    def create_client_socket(self):
        try:
            self.client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.client_socket.bind((self.my_ip, self.my_port))
            self.client_socket.connect((self.ip_to_conn, self.port_to_conn))
            print(f'[+] Initialized socket at {(self.my_ip, self.my_port)} and connected with ({self.ip_to_conn}, {self.port_to_conn})')
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
        return torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')