import socket

def udp_client(message, server_ip, server_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode(), (server_ip, server_port))


udp_client("Hello, Server!", '127.0.0.1', 2002)
