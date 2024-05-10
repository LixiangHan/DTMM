# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class PadV2Options(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PadV2Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsPadV2Options(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def PadV2OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # PadV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def PadV2OptionsStart(builder): builder.StartObject(0)
def Start(builder):
    return PadV2OptionsStart(builder)
def PadV2OptionsEnd(builder): return builder.EndObject()
def End(builder):
    return PadV2OptionsEnd(builder)

class PadV2OptionsT(object):

    # PadV2OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        padV2options = PadV2Options()
        padV2options.Init(buf, pos)
        return cls.InitFromObj(padV2options)

    @classmethod
    def InitFromObj(cls, padV2options):
        x = PadV2OptionsT()
        x._UnPack(padV2options)
        return x

    # PadV2OptionsT
    def _UnPack(self, padV2options):
        if padV2options is None:
            return

    # PadV2OptionsT
    def Pack(self, builder):
        PadV2OptionsStart(builder)
        padV2options = PadV2OptionsEnd(builder)
        return padV2options
