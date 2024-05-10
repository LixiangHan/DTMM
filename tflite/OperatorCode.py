# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class OperatorCode(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OperatorCode()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsOperatorCode(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def OperatorCodeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # OperatorCode
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OperatorCode
    def DeprecatedBuiltinCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # OperatorCode
    def CustomCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # OperatorCode
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # OperatorCode
    def BuiltinCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def OperatorCodeStart(builder): builder.StartObject(4)
def Start(builder):
    return OperatorCodeStart(builder)
def OperatorCodeAddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode): builder.PrependInt8Slot(0, deprecatedBuiltinCode, 0)
def AddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode):
    return OperatorCodeAddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode)
def OperatorCodeAddCustomCode(builder, customCode): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(customCode), 0)
def AddCustomCode(builder, customCode):
    return OperatorCodeAddCustomCode(builder, customCode)
def OperatorCodeAddVersion(builder, version): builder.PrependInt32Slot(2, version, 1)
def AddVersion(builder, version):
    return OperatorCodeAddVersion(builder, version)
def OperatorCodeAddBuiltinCode(builder, builtinCode): builder.PrependInt32Slot(3, builtinCode, 0)
def AddBuiltinCode(builder, builtinCode):
    return OperatorCodeAddBuiltinCode(builder, builtinCode)
def OperatorCodeEnd(builder): return builder.EndObject()
def End(builder):
    return OperatorCodeEnd(builder)

class OperatorCodeT(object):

    # OperatorCodeT
    def __init__(self):
        self.deprecatedBuiltinCode = 0  # type: int
        self.customCode = None  # type: str
        self.version = 1  # type: int
        self.builtinCode = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        operatorCode = OperatorCode()
        operatorCode.Init(buf, pos)
        return cls.InitFromObj(operatorCode)

    @classmethod
    def InitFromObj(cls, operatorCode):
        x = OperatorCodeT()
        x._UnPack(operatorCode)
        return x

    # OperatorCodeT
    def _UnPack(self, operatorCode):
        if operatorCode is None:
            return
        self.deprecatedBuiltinCode = operatorCode.DeprecatedBuiltinCode()
        self.customCode = operatorCode.CustomCode()
        self.version = operatorCode.Version()
        self.builtinCode = operatorCode.BuiltinCode()

    # OperatorCodeT
    def Pack(self, builder):
        if self.customCode is not None:
            customCode = builder.CreateString(self.customCode)
        OperatorCodeStart(builder)
        OperatorCodeAddDeprecatedBuiltinCode(builder, self.deprecatedBuiltinCode)
        if self.customCode is not None:
            OperatorCodeAddCustomCode(builder, customCode)
        OperatorCodeAddVersion(builder, self.version)
        OperatorCodeAddBuiltinCode(builder, self.builtinCode)
        operatorCode = OperatorCodeEnd(builder)
        return operatorCode