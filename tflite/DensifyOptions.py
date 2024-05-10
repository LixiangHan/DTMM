# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DensifyOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DensifyOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDensifyOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DensifyOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # DensifyOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def DensifyOptionsStart(builder): builder.StartObject(0)
def Start(builder):
    return DensifyOptionsStart(builder)
def DensifyOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return DensifyOptionsEnd(builder)

class DensifyOptionsT(object):

    # DensifyOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        densifyOptions = DensifyOptions()
        densifyOptions.Init(buf, pos)
        return cls.InitFromObj(densifyOptions)

    @classmethod
    def InitFromObj(cls, densifyOptions):
        x = DensifyOptionsT()
        x._UnPack(densifyOptions)
        return x

    # DensifyOptionsT
    def _UnPack(self, densifyOptions):
        if densifyOptions is None:
            return

    # DensifyOptionsT
    def Pack(self, builder):
        DensifyOptionsStart(builder)
        densifyOptions = DensifyOptionsEnd(builder)
        return densifyOptions
