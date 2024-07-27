from client import ClientTemplate 
import pickle
import pandas as pd
import argparse
import threading
import time
from client_model import WeakClientModel
import torch

BYTE_CHUNK = 4096
# Threading Events for asynchronous execution
GRADS_RECVD = threading.Event()
AVG_MODEL_RECVD = threading.Event()

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
            elif header == 'avg_model':
                self.client_model.load_state_dict(data[header])
                print(f"\t[+] Received and loaded average model")
                AVG_MODEL_RECVD.set()
        if 'grads' in headers:
            GRADS_RECVD.set()


    def train_one_epoch():
        raise NotImplementedError("Subclasses should implement this method")
        # Implement different functionality

    def train(self, epochs, train_dl, optimizer, fedavg):
        self.client_model.train()
        self.client_model.to(self.device)
        for e in range(epochs):
            print(f"[+] Epoch {e + 1} started")
            for i, (inputs, labels) in enumerate(train_dl):
                inputs, labels = inputs.to(self.device) , labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.client_model(inputs)
                weak_client_outputs = outputs.clone().detach().requires_grad_(True)
                self.send_data_packet(payload={'outputs': weak_client_outputs.requires_grad_(True), 'labels': labels}, comm_socket=self.client_socket)

                GRADS_RECVD.wait()
                GRADS_RECVD.clear()
                #print(self.grads)
                outputs.backward(self.grads)
                optimizer.step()
                #print('took a step')
                
            self.send_data_packet(payload={'epoch_weights': self.client_model.state_dict()}, comm_socket=self.client_socket)
            print(f'\tAverage Training Loss: {(self.curr_loss / len(train_dl)) :.2f}')
            self.curr_loss = 0
            if (e + 1) % fedavg == 0:
                AVG_MODEL_RECVD.wait()
                AVG_MODEL_RECVD.clear()
            time.sleep(1)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', default='127.0.0.1', help='ip of current machine', type=str)
    parser.add_argument('-cp', '--clientport', help='port to connect with the strong client', type=int)       
    parser.add_argument('-ip2con', '--ip2connect', default='127.0.0.1', help='ip of server', type=str)
    parser.add_argument('-p2con', '--port2connect', help='strong client port that accepts connections', type=int) 
    parser.add_argument('-d', '--device', default=None, help='available device to be used', type=str)  
    parser.add_argument('-data', '--datapath', help='path to a data subset', type=str)
    parser.add_argument('-bs', '--batchsize', help='size of batch', type=int) 
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int)
    parser.add_argument('-lr', '--learningrate', help='learning rate', type=float)
    parser.add_argument('-fed', '--fedavg', help='number of epochs', type=int)
    return parser   


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    weak_client = WeakClient(my_ip=args.ip, my_port=args.clientport, ip_to_conn=args.ip2connect, port_to_conn=args.port2connect)
    weak_client.device = torch.device(args.device) if args.device is not None else weak_client.device
    print(f'[+] Using {weak_client.device} as device.')
    weak_client.create_client_socket(client_ip=weak_client.my_ip, client_port=weak_client.my_port, server_ip=weak_client.ip_to_conn, server_port=weak_client.port_to_conn)
    threading.Thread(target=weak_client.listen_for_client_sock_messages, args=()).start()

    #weak_client.device = torch.device('cpu')
    train_dl = weak_client.load_data(subset_path=args.datapath, batch_size=args.batchsize, shuffle=True, num_workers=2)
    weak_client.send_data_packet(payload={'device': weak_client.device, 'datasize': weak_client.datasize}, comm_socket=weak_client.client_socket)

    optimizer = torch.optim.SGD(params=weak_client.client_model.parameters(), lr=args.learningrate)
    
    weak_client.train(epochs=args.epochs, train_dl=train_dl, optimizer=optimizer, fedavg=args.fedavg)