# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReverseV2Options(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReverseV2Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReverseV2Options(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ReverseV2OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ReverseV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def ReverseV2OptionsStart(builder): builder.StartObject(0)
def Start(builder):
    return ReverseV2OptionsStart(builder)
def ReverseV2OptionsEnd(builder): return builder.EndObject()
def End(builder):
    return ReverseV2OptionsEnd(builder)

class ReverseV2OptionsT(object):

    # ReverseV2OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reverseV2options = ReverseV2Options()
        reverseV2options.Init(buf, pos)
        return cls.InitFromObj(reverseV2options)

    @classmethod
    def InitFromObj(cls, reverseV2options):
        x = ReverseV2OptionsT()
        x._UnPack(reverseV2options)
        return x

    # ReverseV2OptionsT
    def _UnPack(self, reverseV2options):
        if reverseV2options is None:
            return

    # ReverseV2OptionsT
    def Pack(self, builder):
        ReverseV2OptionsStart(builder)
        reverseV2options = ReverseV2OptionsEnd(builder)
        return reverseV2options
