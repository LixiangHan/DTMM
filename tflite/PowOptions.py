# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class PowOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PowOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsPowOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def PowOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # PowOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def PowOptionsStart(builder): builder.StartObject(0)
def Start(builder):
    return PowOptionsStart(builder)
def PowOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return PowOptionsEnd(builder)

class PowOptionsT(object):

    # PowOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        powOptions = PowOptions()
        powOptions.Init(buf, pos)
        return cls.InitFromObj(powOptions)

    @classmethod
    def InitFromObj(cls, powOptions):
        x = PowOptionsT()
        x._UnPack(powOptions)
        return x

    # PowOptionsT
    def _UnPack(self, powOptions):
        if powOptions is None:
            return

    # PowOptionsT
    def Pack(self, builder):
        PowOptionsStart(builder)
        powOptions = PowOptionsEnd(builder)
        return powOptions
