# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class HashtableSizeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = HashtableSizeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsHashtableSizeOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def HashtableSizeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # HashtableSizeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def HashtableSizeOptionsStart(builder): builder.StartObject(0)
def Start(builder):
    return HashtableSizeOptionsStart(builder)
def HashtableSizeOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return HashtableSizeOptionsEnd(builder)

class HashtableSizeOptionsT(object):

    # HashtableSizeOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        hashtableSizeOptions = HashtableSizeOptions()
        hashtableSizeOptions.Init(buf, pos)
        return cls.InitFromObj(hashtableSizeOptions)

    @classmethod
    def InitFromObj(cls, hashtableSizeOptions):
        x = HashtableSizeOptionsT()
        x._UnPack(hashtableSizeOptions)
        return x

    # HashtableSizeOptionsT
    def _UnPack(self, hashtableSizeOptions):
        if hashtableSizeOptions is None:
            return

    # HashtableSizeOptionsT
    def Pack(self, builder):
        HashtableSizeOptionsStart(builder)
        hashtableSizeOptions = HashtableSizeOptionsEnd(builder)
        return hashtableSizeOptions
