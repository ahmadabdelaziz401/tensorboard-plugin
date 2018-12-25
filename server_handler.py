import socketserver
import json
from multiprocessing import Process

class TCPHandler(socketserver.StreamRequestHandler):
    def __init__(self, *argv, **kwargs):
        self.dataStruct = []
        super(TCPHandler, self).__init__(*argv, **kwargs)
    def handle(self):
        try:
            self.data = self.rfile.readline().strip()
            print(str(self.data, "utf-8"))
            self.json = json.loads(str(self.data, "utf-8"))


            if self.json["command"] == "stop":
                print("stop")
                self.stop()
                message = "{status: ok}"
                message_bytes = bytes(message, "utf-8")
                self.wfile.write(message_bytes)
                self.server.shutdown()

            elif self.json["command"] == "start":
                print("start")
                self.start()
                message = "{status: ok}"
                message_bytes = bytes(message, "utf-8")
                self.wfile.write(message_bytes)
            elif self.json["command"] == "pause":
                print("pause")
                self.pause()
                message = "{status: ok}"
                message_bytes = bytes(message, "utf-8")
                self.wfile.write(message_bytes)
            elif self.json["command"] == "getdata":
                self.getData()
                message = json.dumps(self.dataStruct)
                print(message)
                self.wfile.write(bytes(message, "utf-8"))
                # Read file, get all datapoints after last known, serve
            else:
                print("ERROR", self.json)
                message = "{status: error}"
                message_bytes = bytes(message, "utf-8")
                self.wfile.write(message_bytes)
        except Exception as e:
            print("Invalid json ", e)
    def getData(self):
        f = open("./progress", "r")
        try:
            allData = json.loads(f.read())
        except Exception as e:
            print("Could not load json ")
            return []
        res = []
        if len(self.dataStruct) == 0:
            if len(allData) != 0:
                self.dataStruct += allData
                res = self.dataStruct
        else:
            for i in range(len(self.dataStruct), len(allData)):
                self.dataStruct += [allData[i]]
                res += [allData[i]]
        f.close()
        return res

    def stop(self):
        self.server.process.terminate()

    def pause(self):
        self.server.process.terminate()

    def start(self):
        self.server.process = Process(target=self.server.instance.main, args=(self.server.instance.log_path,))
        self.server.process.start()
