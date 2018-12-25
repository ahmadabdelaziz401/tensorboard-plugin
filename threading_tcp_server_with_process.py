import socketserver
from multiprocessing import Process


class ThreadingTCPServerWithProcess(socketserver.ThreadingTCPServer):
    def __init__(self, process: Process, instance, server_address, RequestHandler, bind_and_activate: bool = True):
        super().__init__(server_address, RequestHandler, bind_and_activate)
        self.process = process
        self.instance = instance
        self.process.start()

