# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class NegOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NegOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsNegOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def NegOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # NegOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def NegOptionsStart(builder): builder.StartObject(0)
def Start(builder):
    return NegOptionsStart(builder)
def NegOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return NegOptionsEnd(builder)

class NegOptionsT(object):

    # NegOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        negOptions = NegOptions()
        negOptions.Init(buf, pos)
        return cls.InitFromObj(negOptions)

    @classmethod
    def InitFromObj(cls, negOptions):
        x = NegOptionsT()
        x._UnPack(negOptions)
        return x

    # NegOptionsT
    def _UnPack(self, negOptions):
        if negOptions is None:
            return

    # NegOptionsT
    def Pack(self, builder):
        NegOptionsStart(builder)
        negOptions = NegOptionsEnd(builder)
        return negOptions
