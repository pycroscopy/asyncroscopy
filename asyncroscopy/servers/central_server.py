# central_server.py
from twisted.internet.protocol import Protocol, Factory
from twisted.internet import reactor
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.protocols.basic import Int32StringReceiver
from twisted.internet.defer import Deferred
import struct

from asyncroscopy.servers.protocols.central_protocol import CentralProtocol

# Define backend server addresses
# routing_table = {"AS": ("localhost", 9001),
#                 "Gatan": ("localhost", 9002),
#                 "Ceos": ("localhost", 9003),
#                 "Preacquired_AS": ("localhost", 9004)}

routing_table = {"AS": ("10.46.217.241", 9095),
                "Gatan": ("localhost", 9002),
                "Ceos": ("localhost", 9003)}

class CentralFactory(Factory):
    def buildProtocol(self, addr):
        return CentralProtocol(routing_table=routing_table)

# Start the central server
print("Central server running on port 9000...")
reactor.listenTCP(9000, CentralFactory())
reactor.run()