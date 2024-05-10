# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class LSHProjectionOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LSHProjectionOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsLSHProjectionOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def LSHProjectionOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # LSHProjectionOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LSHProjectionOptions
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def LSHProjectionOptionsStart(builder): builder.StartObject(1)
def Start(builder):
    return LSHProjectionOptionsStart(builder)
def LSHProjectionOptionsAddType(builder, type): builder.PrependInt8Slot(0, type, 0)
def AddType(builder, type):
    return LSHProjectionOptionsAddType(builder, type)
def LSHProjectionOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return LSHProjectionOptionsEnd(builder)

class LSHProjectionOptionsT(object):

    # LSHProjectionOptionsT
    def __init__(self):
        self.type = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lshprojectionOptions = LSHProjectionOptions()
        lshprojectionOptions.Init(buf, pos)
        return cls.InitFromObj(lshprojectionOptions)

    @classmethod
    def InitFromObj(cls, lshprojectionOptions):
        x = LSHProjectionOptionsT()
        x._UnPack(lshprojectionOptions)
        return x

    # LSHProjectionOptionsT
    def _UnPack(self, lshprojectionOptions):
        if lshprojectionOptions is None:
            return
        self.type = lshprojectionOptions.Type()

    # LSHProjectionOptionsT
    def Pack(self, builder):
        LSHProjectionOptionsStart(builder)
        LSHProjectionOptionsAddType(builder, self.type)
        lshprojectionOptions = LSHProjectionOptionsEnd(builder)
        return lshprojectionOptions
