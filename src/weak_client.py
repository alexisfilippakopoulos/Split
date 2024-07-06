from client import ClientTemplate 
import pickle
import socket
import sys
import threading
import time
from client_model import WeakClientModel
import torch

BYTE_CHUNK = 4096
# Threading Events for asynchronous execution
GRADS_RECVD = threading.Event()

class WeakClient(ClientTemplate):
    def __init__(self, my_ip, my_port, ip_to_conn, port_to_conn) -> None:
        super().__init__()
        self.my_ip = my_ip
        self.my_port = my_port
        self.ip_to_conn = ip_to_conn
        self.port_to_conn = port_to_conn
        torch.manual_seed(32)
        self.client_model = WeakClientModel()
        self.event_dict = {}
        self.grads = None
        self.curr_loss = 0

    def handle_client_sock_packet(self, data_packet):
        data = pickle.loads(data_packet.split(b'<START>')[1].split(b'<END>')[0])
        headers = list(data.keys())
        for header in headers:
            if header == 'grads':
                self.grads = data[header]
            elif header == 'loss':
                self.curr_loss += data[header]
        if 'grads' in headers:
            GRADS_RECVD.set()


    def train_one_epoch():
        raise NotImplementedError("Subclasses should implement this method")
        # Implement different functionality

    def train(self, epochs, train_dl, optimizer):
        self.client_model.train()
        self.client_model.to(self.device)
        for i in range(epochs):
            print(f"[+] Epoch {i + 1} started")
            for i, (inputs, labels) in enumerate(train_dl):
                inputs, labels = inputs.to(self.device) , labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.client_model(inputs)
                weak_client_outputs = outputs.clone().detach().requires_grad_(True)

                self.send_data_packet(payload={'outputs': weak_client_outputs, 'labels': labels}, comm_socket=self.client_socket)

                GRADS_RECVD.wait()
                GRADS_RECVD.clear()
                #print(self.grads.shape)
                outputs.backward(self.grads)

                optimizer.step()
                #print('Updated weights')
            print('Average Loss:', self.curr_loss / len(train_dl))
            self.curr_loss = 0




if __name__ == '__main__':
    weak_client = WeakClient(my_ip='localhost', my_port=9998, ip_to_conn='localhost', port_to_conn=6969)
    weak_client.create_client_socket(client_ip=weak_client.my_ip, client_port=weak_client.my_port, server_ip=weak_client.ip_to_conn, server_port=weak_client.port_to_conn)
    threading.Thread(target=weak_client.listen_for_client_sock_messages, args=()).start()

    train_dl, datasize = weak_client.load_data(subset_path='subset_data/subset_3_classes.pth', batch_size=32, shuffle=True, num_workers=2)
    weak_client.send_data_packet(payload={'device': weak_client.device, 'datasize': datasize}, comm_socket=weak_client.client_socket)

    optimizer = torch.optim.SGD(params=weak_client.client_model.parameters(), lr=0.01)
    
    weak_client.train(epochs=20, train_dl=train_dl, optimizer=optimizer)