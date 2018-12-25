import socket
import sys
import json
import argparse


class Client:
    parser = argparse.ArgumentParser(add_help=True, description='Benchmarking Client')

    parser.add_argument('--host', action="store", help="address of server, default: localhost", nargs="?",
                        metavar="address", default="localhost")

    parser.add_argument('--port', action="store", help="port of server, default: 9000", nargs="?",
                        metavar="port", default="9000")
    args = parser.parse_args()

    def __init__(self, address=args.host, port=int(args.port)):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = (address, port)
        self.sock.connect(self.server_address)


    def send(self, message="{}"):
        print('connecting to %s port %s' % self.server_address, file=sys.stdout)

        try:
            print('sending "%s"' % message, file=sys.stdout)
            json_message = json.dumps(message)
            byte_stream = bytes(json_message, 'utf-8')
            new_line = bytes("\n", 'utf-8')

            self.sock.sendall(byte_stream)
            self.sock.sendall(new_line)

            response = ""
            while True:
                print("waiting for response")
                data = self.sock.recv(1024)
                print("received some response")
                response += data.decode("utf-8")
                if not data: break

            print('received "%s"' % response, file=sys.stdout)
            return response

        finally:
            # print ( 'closing socket', file=sys.stdout)
            self.sock.close()



c = Client()

c.send('{"command": "pause"}')
c.send('{"command": "start"}')
c.send('{"command": "stop"}')
