# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class WhileOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WhileOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsWhileOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def WhileOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # WhileOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # WhileOptions
    def CondSubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # WhileOptions
    def BodySubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def WhileOptionsStart(builder): builder.StartObject(2)
def Start(builder):
    return WhileOptionsStart(builder)
def WhileOptionsAddCondSubgraphIndex(builder, condSubgraphIndex): builder.PrependInt32Slot(0, condSubgraphIndex, 0)
def AddCondSubgraphIndex(builder, condSubgraphIndex):
    return WhileOptionsAddCondSubgraphIndex(builder, condSubgraphIndex)
def WhileOptionsAddBodySubgraphIndex(builder, bodySubgraphIndex): builder.PrependInt32Slot(1, bodySubgraphIndex, 0)
def AddBodySubgraphIndex(builder, bodySubgraphIndex):
    return WhileOptionsAddBodySubgraphIndex(builder, bodySubgraphIndex)
def WhileOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return WhileOptionsEnd(builder)

class WhileOptionsT(object):

    # WhileOptionsT
    def __init__(self):
        self.condSubgraphIndex = 0  # type: int
        self.bodySubgraphIndex = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        whileOptions = WhileOptions()
        whileOptions.Init(buf, pos)
        return cls.InitFromObj(whileOptions)

    @classmethod
    def InitFromObj(cls, whileOptions):
        x = WhileOptionsT()
        x._UnPack(whileOptions)
        return x

    # WhileOptionsT
    def _UnPack(self, whileOptions):
        if whileOptions is None:
            return
        self.condSubgraphIndex = whileOptions.CondSubgraphIndex()
        self.bodySubgraphIndex = whileOptions.BodySubgraphIndex()

    # WhileOptionsT
    def Pack(self, builder):
        WhileOptionsStart(builder)
        WhileOptionsAddCondSubgraphIndex(builder, self.condSubgraphIndex)
        WhileOptionsAddBodySubgraphIndex(builder, self.bodySubgraphIndex)
        whileOptions = WhileOptionsEnd(builder)
        return whileOptions
