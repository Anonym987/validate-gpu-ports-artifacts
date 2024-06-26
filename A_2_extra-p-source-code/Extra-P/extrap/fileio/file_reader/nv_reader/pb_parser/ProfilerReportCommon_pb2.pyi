# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.internal.enum_type_wrapper import (
    EnumTypeWrapper as google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    NewType as typing___NewType,
    Optional as typing___Optional,
    Text as typing___Text,
    cast as typing___cast,
)

from typing import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

APITypeValue = typing___NewType('APITypeValue', builtin___int)
type___APITypeValue = APITypeValue
APIType: _APIType
class _APIType(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[APITypeValue]):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    APIType_CUDA = typing___cast(APITypeValue, 0)
    APIType_OpenCL = typing___cast(APITypeValue, 1)
    APIType_Direct3D = typing___cast(APITypeValue, 2)
    APIType_OpenGL = typing___cast(APITypeValue, 3)
APIType_CUDA = typing___cast(APITypeValue, 0)
APIType_OpenCL = typing___cast(APITypeValue, 1)
APIType_Direct3D = typing___cast(APITypeValue, 2)
APIType_OpenGL = typing___cast(APITypeValue, 3)
type___APIType = APIType

SourceSassLevelValue = typing___NewType('SourceSassLevelValue', builtin___int)
type___SourceSassLevelValue = SourceSassLevelValue
SourceSassLevel: _SourceSassLevel
class _SourceSassLevel(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[SourceSassLevelValue]):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    SourceSassLevel_Unset = typing___cast(SourceSassLevelValue, 0)
    SourceSassLevel_Sass1 = typing___cast(SourceSassLevelValue, 1)
    SourceSassLevel_Sass2 = typing___cast(SourceSassLevelValue, 2)
    SourceSassLevel_Sass3 = typing___cast(SourceSassLevelValue, 3)
    SourceSassLevel_Sass4 = typing___cast(SourceSassLevelValue, 4)
    SourceSassLevel_Sass5 = typing___cast(SourceSassLevelValue, 5)
    SourceSassLevel_Sass6 = typing___cast(SourceSassLevelValue, 6)
    SourceSassLevel_Sass7 = typing___cast(SourceSassLevelValue, 7)
SourceSassLevel_Unset = typing___cast(SourceSassLevelValue, 0)
SourceSassLevel_Sass1 = typing___cast(SourceSassLevelValue, 1)
SourceSassLevel_Sass2 = typing___cast(SourceSassLevelValue, 2)
SourceSassLevel_Sass3 = typing___cast(SourceSassLevelValue, 3)
SourceSassLevel_Sass4 = typing___cast(SourceSassLevelValue, 4)
SourceSassLevel_Sass5 = typing___cast(SourceSassLevelValue, 5)
SourceSassLevel_Sass6 = typing___cast(SourceSassLevelValue, 6)
SourceSassLevel_Sass7 = typing___cast(SourceSassLevelValue, 7)
type___SourceSassLevel = SourceSassLevel

class Uint64x3(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    X: builtin___int = ...
    Y: builtin___int = ...
    Z: builtin___int = ...

    def __init__(self,
        *,
        X : typing___Optional[builtin___int] = None,
        Y : typing___Optional[builtin___int] = None,
        Z : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"X",b"X",u"Y",b"Y",u"Z",b"Z"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"X",b"X",u"Y",b"Y",u"Z",b"Z"]) -> None: ...
type___Uint64x3 = Uint64x3

class SourceData(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Reference: builtin___int = ...
    Code: typing___Text = ...
    Intermediate: builtin___bytes = ...
    Binary: builtin___bytes = ...
    SassLevel: type___SourceSassLevelValue = ...
    SMRevision: builtin___int = ...
    BinaryFlags: builtin___int = ...

    def __init__(self,
        *,
        Reference : typing___Optional[builtin___int] = None,
        Code : typing___Optional[typing___Text] = None,
        Intermediate : typing___Optional[builtin___bytes] = None,
        Binary : typing___Optional[builtin___bytes] = None,
        SassLevel : typing___Optional[type___SourceSassLevelValue] = None,
        SMRevision : typing___Optional[builtin___int] = None,
        BinaryFlags : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Binary",b"Binary",u"BinaryFlags",b"BinaryFlags",u"Code",b"Code",u"Intermediate",b"Intermediate",u"Reference",b"Reference",u"SMRevision",b"SMRevision",u"SassLevel",b"SassLevel"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Binary",b"Binary",u"BinaryFlags",b"BinaryFlags",u"Code",b"Code",u"Intermediate",b"Intermediate",u"Reference",b"Reference",u"SMRevision",b"SMRevision",u"SassLevel",b"SassLevel"]) -> None: ...
type___SourceData = SourceData

class ExecutableSettings(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    ExecutablePath: typing___Text = ...
    WorkDirectory: typing___Text = ...
    CmdlineAgruments: typing___Text = ...
    Environment: typing___Text = ...

    def __init__(self,
        *,
        ExecutablePath : typing___Optional[typing___Text] = None,
        WorkDirectory : typing___Optional[typing___Text] = None,
        CmdlineAgruments : typing___Optional[typing___Text] = None,
        Environment : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"CmdlineAgruments",b"CmdlineAgruments",u"Environment",b"Environment",u"ExecutablePath",b"ExecutablePath",u"WorkDirectory",b"WorkDirectory"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"CmdlineAgruments",b"CmdlineAgruments",u"Environment",b"Environment",u"ExecutablePath",b"ExecutablePath",u"WorkDirectory",b"WorkDirectory"]) -> None: ...
type___ExecutableSettings = ExecutableSettings
