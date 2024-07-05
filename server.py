import socket
import threading
import pickle
import sys

BYTE_CHUNK = 4096

class Server():
    def __init__(self, my_ip, my_port) -> None:
        self.server_ip = my_ip
        self.server_port = my_port

    def create_server_socket(self):
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.server_socket.bind((self.server_ip, self.server_port))
        print(f'[+] Server initialized successfully at {self.server_ip, self.server_port}')\

    def listen_for_connections(self):
        self.server_socket.listen()
        while True:
            client_socket, client_address = self.server_socket.accept()
            print(f"[+] Client {client_address} connected")
            threading.Thread(target=self.listen_for_messages, args=(client_socket, client_address)).start()

    def listen_for_messages(self, client_socket, client_address):
        # Communication thread with each weak client
        self.send_data('hiii strong client', client_socket)
        data_packet = b''
        while True:
            data_chunk = client_socket.recv(BYTE_CHUNK)
            data_packet += data_chunk
            if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        print('aaa')
                        threading.Thread(target=self.handle_data_packet, args=(data_packet, client_socket)).start()
                        data_packet = b'' 
            if not data_chunk:
                break

    def handle_data_packet(self, data_packet, client_socket):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        print(data)

    def send_data(self, data, comm_socket):
        comm_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')



if __name__ == '__main__':
    server = Server(my_ip='localhost', my_port=10000)
    server.create_server_socket()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    

