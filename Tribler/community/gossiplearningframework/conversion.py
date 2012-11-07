from struct import pack, unpack_from
import copy
import json
import numpy

from Tribler.dispersy.message import DropPacket
from Tribler.dispersy.conversion import BinaryConversion
from Tribler.community.gossiplearningframework.payload import *
# TODO: autoload
from Tribler.community.gossiplearningframework.models.logisticregression import LogisticRegressionModel
from Tribler.community.gossiplearningframework.models.adalineperceptron import AdalinePerceptronModel
from Tribler.community.gossiplearningframework.models.p2pegasos import P2PegasosModel

if __debug__:
    from Tribler.dispersy.dprint import dprint

class JSONConversion(BinaryConversion):
    def __init__(self, community):
        super(JSONConversion, self).__init__(community, "\x01") # Community version 1
        # Message type ID 1
        self.define_meta_message(chr(1), community.get_meta_message(u"modeldata"), self._encode_json, self._decode_json)

    def _encode_json(self, message):
        dprint(type(message.payload.message))
        dprint(message.payload.message)
        assert isinstance(message.payload.message, GossipMessage)
        wiredata = json.dumps(message.payload.message, cls=ClassCoder).encode("UTF-8")

        assert len(wiredata) < 2**16

        if __debug__: dprint(wiredata)

        # Encode the length on 2 bytes, network byte order. The wire data follows.
        return pack("!H", len(wiredata)), wiredata

    def _decode_json(self, meta_message, offset, data):
        if len(data) < offset + 2:
            raise DropPacket("Insufficient packet size")

        data_length, = unpack_from("!H", data, offset)
        offset += 2

        try:
            wiredata = json.loads(data[offset:offset + data_length].decode("UTF-8"),
                                  object_hook=ClassCoder.decode_class)
            offset += data_length
        except UnicodeError:
            raise DropPacket("Unable to decode UTF-8")

        return offset, meta_message.meta.payload.implement(wiredata)

class ClassCoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GossipMessage):
            # Get a copy of class variables.
            result = copy.deepcopy(obj.__dict__)

            # Add the class name.
            result[u'__class__'] = obj.__class__.__name__
            return result
        
        elif isinstance(obj, numpy.ndarray):
            return {'__class__':'ndarray', 'value': obj.tolist()}
        
        else:
            return json.JSONEncoder.default(self, obj)

    @classmethod
    def decode_class(cls, d):
        if isinstance(d, dict) and '__class__' in d:
            if d['__class__'] in ['GossipMessage', 'AdalinePerceptronModel', 'LogisticRegressionModel', 'P2PegasosModel', ]:
                # Get the class, create object.
                res = globals()[str(d['__class__'])]()
    
                # Update class variables recursively.
                for k, v in d.items():
                    if k != '__class__':
                        res.__dict__[k] = cls.decode_class(v)
    
                return res
            elif d['__class__'] == 'ndarray':
                return numpy.array(d['value'])
        return d

