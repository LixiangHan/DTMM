# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Conv3DOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Conv3DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsConv3DOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def Conv3DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Conv3DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Conv3DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideD(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def DilationDFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

def Conv3DOptionsStart(builder): builder.StartObject(8)
def Start(builder):
    return Conv3DOptionsStart(builder)
def Conv3DOptionsAddPadding(builder, padding): builder.PrependInt8Slot(0, padding, 0)
def AddPadding(builder, padding):
    return Conv3DOptionsAddPadding(builder, padding)
def Conv3DOptionsAddStrideD(builder, strideD): builder.PrependInt32Slot(1, strideD, 0)
def AddStrideD(builder, strideD):
    return Conv3DOptionsAddStrideD(builder, strideD)
def Conv3DOptionsAddStrideW(builder, strideW): builder.PrependInt32Slot(2, strideW, 0)
def AddStrideW(builder, strideW):
    return Conv3DOptionsAddStrideW(builder, strideW)
def Conv3DOptionsAddStrideH(builder, strideH): builder.PrependInt32Slot(3, strideH, 0)
def AddStrideH(builder, strideH):
    return Conv3DOptionsAddStrideH(builder, strideH)
def Conv3DOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(4, fusedActivationFunction, 0)
def AddFusedActivationFunction(builder, fusedActivationFunction):
    return Conv3DOptionsAddFusedActivationFunction(builder, fusedActivationFunction)
def Conv3DOptionsAddDilationDFactor(builder, dilationDFactor): builder.PrependInt32Slot(5, dilationDFactor, 1)
def AddDilationDFactor(builder, dilationDFactor):
    return Conv3DOptionsAddDilationDFactor(builder, dilationDFactor)
def Conv3DOptionsAddDilationWFactor(builder, dilationWFactor): builder.PrependInt32Slot(6, dilationWFactor, 1)
def AddDilationWFactor(builder, dilationWFactor):
    return Conv3DOptionsAddDilationWFactor(builder, dilationWFactor)
def Conv3DOptionsAddDilationHFactor(builder, dilationHFactor): builder.PrependInt32Slot(7, dilationHFactor, 1)
def AddDilationHFactor(builder, dilationHFactor):
    return Conv3DOptionsAddDilationHFactor(builder, dilationHFactor)
def Conv3DOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return Conv3DOptionsEnd(builder)

class Conv3DOptionsT(object):

    # Conv3DOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideD = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int
        self.fusedActivationFunction = 0  # type: int
        self.dilationDFactor = 1  # type: int
        self.dilationWFactor = 1  # type: int
        self.dilationHFactor = 1  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conv3doptions = Conv3DOptions()
        conv3doptions.Init(buf, pos)
        return cls.InitFromObj(conv3doptions)

    @classmethod
    def InitFromObj(cls, conv3doptions):
        x = Conv3DOptionsT()
        x._UnPack(conv3doptions)
        return x

    # Conv3DOptionsT
    def _UnPack(self, conv3doptions):
        if conv3doptions is None:
            return
        self.padding = conv3doptions.Padding()
        self.strideD = conv3doptions.StrideD()
        self.strideW = conv3doptions.StrideW()
        self.strideH = conv3doptions.StrideH()
        self.fusedActivationFunction = conv3doptions.FusedActivationFunction()
        self.dilationDFactor = conv3doptions.DilationDFactor()
        self.dilationWFactor = conv3doptions.DilationWFactor()
        self.dilationHFactor = conv3doptions.DilationHFactor()

    # Conv3DOptionsT
    def Pack(self, builder):
        Conv3DOptionsStart(builder)
        Conv3DOptionsAddPadding(builder, self.padding)
        Conv3DOptionsAddStrideD(builder, self.strideD)
        Conv3DOptionsAddStrideW(builder, self.strideW)
        Conv3DOptionsAddStrideH(builder, self.strideH)
        Conv3DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        Conv3DOptionsAddDilationDFactor(builder, self.dilationDFactor)
        Conv3DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        Conv3DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        conv3doptions = Conv3DOptionsEnd(builder)
        return conv3doptions
