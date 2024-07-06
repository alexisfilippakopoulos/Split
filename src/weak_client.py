from client import ClientTemplate 
import pickle
import socket
import sys
import threading
import time

BYTE_CHUNK = 4096

class WeakClient(ClientTemplate):
    def __init__(self, my_ip, my_port, ip_to_conn, port_to_conn) -> None:
        super().__init__()
        self.my_ip = my_ip
        self.my_port = my_port
        self.ip_to_conn = ip_to_conn
        self.port_to_conn = port_to_conn

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
    weak_client = WeakClient(my_ip='localhost', my_port=9998, ip_to_conn='localhost', port_to_conn=6969)
    weak_client.create_client_socket(client_ip=weak_client.my_ip, client_port=weak_client.my_port, server_ip=weak_client.ip_to_conn, server_port=weak_client.port_to_conn)
    time.sleep(3)
    threading.Thread(target=weak_client.listen_for_client_sock_messages, args=()).start()
    weak_client.send_data_packet('hi strong client', weak_client.client_socket)