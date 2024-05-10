# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class LSTMOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LSTMOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsLSTMOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def LSTMOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # LSTMOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LSTMOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # LSTMOptions
    def CellClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # LSTMOptions
    def ProjClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # LSTMOptions
    def KernelType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # LSTMOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def LSTMOptionsStart(builder): builder.StartObject(5)
def Start(builder):
    return LSTMOptionsStart(builder)
def LSTMOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(0, fusedActivationFunction, 0)
def AddFusedActivationFunction(builder, fusedActivationFunction):
    return LSTMOptionsAddFusedActivationFunction(builder, fusedActivationFunction)
def LSTMOptionsAddCellClip(builder, cellClip): builder.PrependFloat32Slot(1, cellClip, 0.0)
def AddCellClip(builder, cellClip):
    return LSTMOptionsAddCellClip(builder, cellClip)
def LSTMOptionsAddProjClip(builder, projClip): builder.PrependFloat32Slot(2, projClip, 0.0)
def AddProjClip(builder, projClip):
    return LSTMOptionsAddProjClip(builder, projClip)
def LSTMOptionsAddKernelType(builder, kernelType): builder.PrependInt8Slot(3, kernelType, 0)
def AddKernelType(builder, kernelType):
    return LSTMOptionsAddKernelType(builder, kernelType)
def LSTMOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs): builder.PrependBoolSlot(4, asymmetricQuantizeInputs, 0)
def AddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    return LSTMOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs)
def LSTMOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return LSTMOptionsEnd(builder)

class LSTMOptionsT(object):

    # LSTMOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.cellClip = 0.0  # type: float
        self.projClip = 0.0  # type: float
        self.kernelType = 0  # type: int
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lstmoptions = LSTMOptions()
        lstmoptions.Init(buf, pos)
        return cls.InitFromObj(lstmoptions)

    @classmethod
    def InitFromObj(cls, lstmoptions):
        x = LSTMOptionsT()
        x._UnPack(lstmoptions)
        return x

    # LSTMOptionsT
    def _UnPack(self, lstmoptions):
        if lstmoptions is None:
            return
        self.fusedActivationFunction = lstmoptions.FusedActivationFunction()
        self.cellClip = lstmoptions.CellClip()
        self.projClip = lstmoptions.ProjClip()
        self.kernelType = lstmoptions.KernelType()
        self.asymmetricQuantizeInputs = lstmoptions.AsymmetricQuantizeInputs()

    # LSTMOptionsT
    def Pack(self, builder):
        LSTMOptionsStart(builder)
        LSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        LSTMOptionsAddCellClip(builder, self.cellClip)
        LSTMOptionsAddProjClip(builder, self.projClip)
        LSTMOptionsAddKernelType(builder, self.kernelType)
        LSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        lstmoptions = LSTMOptionsEnd(builder)
        return lstmoptions
