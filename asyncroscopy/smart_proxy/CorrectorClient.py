#!/usr/bin/python
"""
A small client for the CEOS RPC interface. It can be used to demonstrate the
functionality and for testing the implementation of the server.
"""
import logging
import json

from twisted.internet import reactor,defer
from twisted.protocols.basic import NetstringReceiver
from twisted.internet.protocol import ReconnectingClientFactory

logging.basicConfig()
log = logging.getLogger('CEOS_acquisition')
log.setLevel(logging.INFO)


class CEOSProtocol(NetstringReceiver):
    """
    A protocol for twisted implementing JSON-RPC using Netstring for framing.
    See also:
    https://cr.yp.to/proto/netstrings.txt
    http://www.jsonrpc.org/specification
    """

    def __init__(self):
        self._nextMessageID = 1
        self._pendingCommands = {}

    def connectionMade(self):
        """
        Called by the protocol factory after the connection to the server has
        been established.
        """
        log.info('Connected to CEOS server')
        self.factory.setProtocol(self, msg)

    def connectionLost(self, reason):
        """
        Called by twisted after the connection to the server has been
        interrupted.

        :param reason: a twisted Failure instance
        """
        log.info('Client disconnected: %s', reason.getErrorMessage())
        self.factory.gui.setProtocol(None, reason.getErrorMessage())
        for d in self._pendingCommands.values():
            d.errback(reason)
        self._pendingCommands.clear()

    def disconnect(self):
        """
        Disconnect from server.
        """
        self.transport.loseConnection()

    def stringReceived(self, data):
        """
        Called by NetstringReceiver after receiving a complete string.

        :param string: the data as raw string
        """
        data = json.loads(string)

        d =  self._pendingCommands.pop(msg['id'], None)
        if d:
            if 'error' in msg:
                d.errback(Exception(msg["error"]["message"]))
            else:
                d.callback(msg["result"])
                 
    def callCommand(self, name, parameter=None):
        """
        Send a RPC request to the server.

        :param name: name of command
        :param parameter: a dict or list of parameters
        :returns: a Deferred that fires after receiving the reply from the
                  server
        """
        if parameter is None:
            parameter = {}
        log.info('Calling %s params=%s', name, parameter)
        data = {
            'jsonrpc': '2.0',
            'id': self._nextMessageID,
            'method': name,
            'params': parameter}

        d = defer.Deferred()
        self._pendingCommands[self._nextMessageID] = d
        self._nextMessageID += 1

        self.sendString(json.dumps(data).encode('utf-8'))

        return d



class CEOSClient(ReconnectingClientFactory):
    """
    Creates an instance of the NetstringJSONRPCClientProtocol after TCP
    connection to the server has been established.
    Automatically reconnects after connection has been interrupted.
    """
    protocol = CEOSProtocol

    # speed up reconnecting
    maxDelay = 3

    def __init__(self, client):
        self.client = client

