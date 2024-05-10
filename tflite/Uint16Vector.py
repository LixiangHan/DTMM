# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Uint16Vector(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Uint16Vector()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsUint16Vector(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def Uint16VectorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Uint16Vector
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Uint16Vector
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # Uint16Vector
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint16Flags, o)
        return 0

    # Uint16Vector
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Uint16Vector
    def ValuesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def Uint16VectorStart(builder): builder.StartObject(1)
def Start(builder):
    return Uint16VectorStart(builder)
def Uint16VectorAddValues(builder, values): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0)
def AddValues(builder, values):
    return Uint16VectorAddValues(builder, values)
def Uint16VectorStartValuesVector(builder, numElems): return builder.StartVector(2, numElems, 2)
def StartValuesVector(builder, numElems):
    return Uint16VectorStartValuesVector(builder, numElems)
def Uint16VectorEnd(builder): return builder.EndObject()
def End(builder):
    return Uint16VectorEnd(builder)
try:
    from typing import List
except:
    pass

class Uint16VectorT(object):

    # Uint16VectorT
    def __init__(self):
        self.values = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        uint16vector = Uint16Vector()
        uint16vector.Init(buf, pos)
        return cls.InitFromObj(uint16vector)

    @classmethod
    def InitFromObj(cls, uint16vector):
        x = Uint16VectorT()
        x._UnPack(uint16vector)
        return x

    # Uint16VectorT
    def _UnPack(self, uint16vector):
        if uint16vector is None:
            return
        if not uint16vector.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(uint16vector.ValuesLength()):
                    self.values.append(uint16vector.Values(i))
            else:
                self.values = uint16vector.ValuesAsNumpy()

    # Uint16VectorT
    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                Uint16VectorStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependUint16(self.values[i])
                values = builder.EndVector()
        Uint16VectorStart(builder)
        if self.values is not None:
            Uint16VectorAddValues(builder, values)
        uint16vector = Uint16VectorEnd(builder)
        return uint16vector