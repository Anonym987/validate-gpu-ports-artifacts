# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from .ProfilerMetricOptions_pb2 import (
    MetricOptionFilter as ProfilerMetricOptions_pb2___MetricOptionFilter,
)

from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
)

from google.protobuf.internal.enum_type_wrapper import (
    EnumTypeWrapper as google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
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

HWUnitTypeValue = typing___NewType('HWUnitTypeValue', builtin___int)
type___HWUnitTypeValue = HWUnitTypeValue
HWUnitType: _HWUnitType
class _HWUnitType(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[HWUnitTypeValue]):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    Invalid = typing___cast(HWUnitTypeValue, 0)
    Default = typing___cast(HWUnitTypeValue, 1)
    Gpc = typing___cast(HWUnitTypeValue, 2)
    Tpc = typing___cast(HWUnitTypeValue, 3)
    Sm = typing___cast(HWUnitTypeValue, 4)
    Smsp = typing___cast(HWUnitTypeValue, 5)
    Tex = typing___cast(HWUnitTypeValue, 6)
    Lts = typing___cast(HWUnitTypeValue, 7)
    Ltc = typing___cast(HWUnitTypeValue, 8)
    Fbpa = typing___cast(HWUnitTypeValue, 9)
Invalid = typing___cast(HWUnitTypeValue, 0)
Default = typing___cast(HWUnitTypeValue, 1)
Gpc = typing___cast(HWUnitTypeValue, 2)
Tpc = typing___cast(HWUnitTypeValue, 3)
Sm = typing___cast(HWUnitTypeValue, 4)
Smsp = typing___cast(HWUnitTypeValue, 5)
Tex = typing___cast(HWUnitTypeValue, 6)
Lts = typing___cast(HWUnitTypeValue, 7)
Ltc = typing___cast(HWUnitTypeValue, 8)
Fbpa = typing___cast(HWUnitTypeValue, 9)
type___HWUnitType = HWUnitType

class ProfilerSectionMetricOption(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Name: typing___Text = ...
    Label: typing___Text = ...

    @property
    def Filter(self) -> ProfilerMetricOptions_pb2___MetricOptionFilter: ...

    def __init__(self,
        *,
        Name : typing___Optional[typing___Text] = None,
        Label : typing___Optional[typing___Text] = None,
        Filter : typing___Optional[ProfilerMetricOptions_pb2___MetricOptionFilter] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Filter",b"Filter",u"Label",b"Label",u"Name",b"Name"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Filter",b"Filter",u"Label",b"Label",u"Name",b"Name"]) -> None: ...
type___ProfilerSectionMetricOption = ProfilerSectionMetricOption

class ProfilerSectionMetric(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Name: typing___Text = ...
    Label: typing___Text = ...
    HWUnit: type___HWUnitTypeValue = ...
    ShowInstances: builtin___bool = ...
    Unit: typing___Text = ...

    @property
    def Filter(self) -> ProfilerMetricOptions_pb2___MetricOptionFilter: ...

    @property
    def Options(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetricOption]: ...

    def __init__(self,
        *,
        Name : typing___Optional[typing___Text] = None,
        Label : typing___Optional[typing___Text] = None,
        HWUnit : typing___Optional[type___HWUnitTypeValue] = None,
        ShowInstances : typing___Optional[builtin___bool] = None,
        Unit : typing___Optional[typing___Text] = None,
        Filter : typing___Optional[ProfilerMetricOptions_pb2___MetricOptionFilter] = None,
        Options : typing___Optional[typing___Iterable[type___ProfilerSectionMetricOption]] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Filter",b"Filter",u"HWUnit",b"HWUnit",u"Label",b"Label",u"Name",b"Name",u"ShowInstances",b"ShowInstances",u"Unit",b"Unit"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Filter",b"Filter",u"HWUnit",b"HWUnit",u"Label",b"Label",u"Name",b"Name",u"Options",b"Options",u"ShowInstances",b"ShowInstances",u"Unit",b"Unit"]) -> None: ...
type___ProfilerSectionMetric = ProfilerSectionMetric

class ProfilerSectionHighlightX(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def Metrics(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetric]: ...

    def __init__(self,
        *,
        Metrics : typing___Optional[typing___Iterable[type___ProfilerSectionMetric]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Metrics",b"Metrics"]) -> None: ...
type___ProfilerSectionHighlightX = ProfilerSectionHighlightX

class ProfilerSectionTable(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    LayoutOrderValue = typing___NewType('LayoutOrderValue', builtin___int)
    type___LayoutOrderValue = LayoutOrderValue
    LayoutOrder: _LayoutOrder
    class _LayoutOrder(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[ProfilerSectionTable.LayoutOrderValue]):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        RowMajor = typing___cast(ProfilerSectionTable.LayoutOrderValue, 0)
        ColumnMajor = typing___cast(ProfilerSectionTable.LayoutOrderValue, 1)
    RowMajor = typing___cast(ProfilerSectionTable.LayoutOrderValue, 0)
    ColumnMajor = typing___cast(ProfilerSectionTable.LayoutOrderValue, 1)
    type___LayoutOrder = LayoutOrder

    Label: typing___Text = ...
    Rows: builtin___int = ...
    Columns: builtin___int = ...
    Order: type___ProfilerSectionTable.LayoutOrderValue = ...

    @property
    def Metrics(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetric]: ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        Rows : typing___Optional[builtin___int] = None,
        Columns : typing___Optional[builtin___int] = None,
        Order : typing___Optional[type___ProfilerSectionTable.LayoutOrderValue] = None,
        Metrics : typing___Optional[typing___Iterable[type___ProfilerSectionMetric]] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Columns",b"Columns",u"Label",b"Label",u"Order",b"Order",u"Rows",b"Rows"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Columns",b"Columns",u"Label",b"Label",u"Metrics",b"Metrics",u"Order",b"Order",u"Rows",b"Rows"]) -> None: ...
type___ProfilerSectionTable = ProfilerSectionTable

class ProfilerSectionChartAxisRange(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Min: builtin___int = ...
    Max: builtin___int = ...

    def __init__(self,
        *,
        Min : typing___Optional[builtin___int] = None,
        Max : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Max",b"Max",u"Min",b"Min"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Max",b"Max",u"Min",b"Min"]) -> None: ...
type___ProfilerSectionChartAxisRange = ProfilerSectionChartAxisRange

class ProfilerSectionChartValueAxis(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...
    TickCount: builtin___int = ...
    Size: builtin___int = ...
    Precision: builtin___int = ...

    @property
    def Range(self) -> type___ProfilerSectionChartAxisRange: ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        Range : typing___Optional[type___ProfilerSectionChartAxisRange] = None,
        TickCount : typing___Optional[builtin___int] = None,
        Size : typing___Optional[builtin___int] = None,
        Precision : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"Precision",b"Precision",u"Range",b"Range",u"Size",b"Size",u"TickCount",b"TickCount"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"Precision",b"Precision",u"Range",b"Range",u"Size",b"Size",u"TickCount",b"TickCount"]) -> None: ...
type___ProfilerSectionChartValueAxis = ProfilerSectionChartValueAxis

class ProfilerSectionChartCategoryAxis(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label"]) -> None: ...
type___ProfilerSectionChartCategoryAxis = ProfilerSectionChartCategoryAxis

class ProfilerSectionChartHistogramAxis(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...
    BinCount: builtin___int = ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        BinCount : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"BinCount",b"BinCount",u"Label",b"Label"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"BinCount",b"BinCount",u"Label",b"Label"]) -> None: ...
type___ProfilerSectionChartHistogramAxis = ProfilerSectionChartHistogramAxis

class ProfilerSectionBarChart(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    DirectionsValue = typing___NewType('DirectionsValue', builtin___int)
    type___DirectionsValue = DirectionsValue
    Directions: _Directions
    class _Directions(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[ProfilerSectionBarChart.DirectionsValue]):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        Horizontal = typing___cast(ProfilerSectionBarChart.DirectionsValue, 0)
        Vertical = typing___cast(ProfilerSectionBarChart.DirectionsValue, 1)
    Horizontal = typing___cast(ProfilerSectionBarChart.DirectionsValue, 0)
    Vertical = typing___cast(ProfilerSectionBarChart.DirectionsValue, 1)
    type___Directions = Directions

    Label: typing___Text = ...
    Direction: type___ProfilerSectionBarChart.DirectionsValue = ...

    @property
    def CategoryAxis(self) -> type___ProfilerSectionChartCategoryAxis: ...

    @property
    def ValueAxis(self) -> type___ProfilerSectionChartValueAxis: ...

    @property
    def Metrics(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetric]: ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        Direction : typing___Optional[type___ProfilerSectionBarChart.DirectionsValue] = None,
        CategoryAxis : typing___Optional[type___ProfilerSectionChartCategoryAxis] = None,
        ValueAxis : typing___Optional[type___ProfilerSectionChartValueAxis] = None,
        Metrics : typing___Optional[typing___Iterable[type___ProfilerSectionMetric]] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"CategoryAxis",b"CategoryAxis",u"Direction",b"Direction",u"Label",b"Label",u"ValueAxis",b"ValueAxis"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"CategoryAxis",b"CategoryAxis",u"Direction",b"Direction",u"Label",b"Label",u"Metrics",b"Metrics",u"ValueAxis",b"ValueAxis"]) -> None: ...
type___ProfilerSectionBarChart = ProfilerSectionBarChart

class ProfilerSectionHistogramChart(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...

    @property
    def HistogramAxis(self) -> type___ProfilerSectionChartHistogramAxis: ...

    @property
    def ValueAxis(self) -> type___ProfilerSectionChartValueAxis: ...

    @property
    def Metric(self) -> type___ProfilerSectionMetric: ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        HistogramAxis : typing___Optional[type___ProfilerSectionChartHistogramAxis] = None,
        ValueAxis : typing___Optional[type___ProfilerSectionChartValueAxis] = None,
        Metric : typing___Optional[type___ProfilerSectionMetric] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"HistogramAxis",b"HistogramAxis",u"Label",b"Label",u"Metric",b"Metric",u"ValueAxis",b"ValueAxis"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"HistogramAxis",b"HistogramAxis",u"Label",b"Label",u"Metric",b"Metric",u"ValueAxis",b"ValueAxis"]) -> None: ...
type___ProfilerSectionHistogramChart = ProfilerSectionHistogramChart

class ProfilerSectionLineChart(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...

    @property
    def AxisX(self) -> type___ProfilerSectionChartValueAxis: ...

    @property
    def AxisY(self) -> type___ProfilerSectionChartValueAxis: ...

    @property
    def Metrics(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetric]: ...

    @property
    def HighlightX(self) -> type___ProfilerSectionHighlightX: ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        AxisX : typing___Optional[type___ProfilerSectionChartValueAxis] = None,
        AxisY : typing___Optional[type___ProfilerSectionChartValueAxis] = None,
        Metrics : typing___Optional[typing___Iterable[type___ProfilerSectionMetric]] = None,
        HighlightX : typing___Optional[type___ProfilerSectionHighlightX] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"AxisX",b"AxisX",u"AxisY",b"AxisY",u"HighlightX",b"HighlightX",u"Label",b"Label"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"AxisX",b"AxisX",u"AxisY",b"AxisY",u"HighlightX",b"HighlightX",u"Label",b"Label",u"Metrics",b"Metrics"]) -> None: ...
type___ProfilerSectionLineChart = ProfilerSectionLineChart

class ProfilerSectionMemorySharedTable(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...
    ShowLoads: builtin___bool = ...
    ShowStores: builtin___bool = ...
    ShowAtomics: builtin___bool = ...
    ShowTotals: builtin___bool = ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        ShowLoads : typing___Optional[builtin___bool] = None,
        ShowStores : typing___Optional[builtin___bool] = None,
        ShowAtomics : typing___Optional[builtin___bool] = None,
        ShowTotals : typing___Optional[builtin___bool] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowAtomics",b"ShowAtomics",u"ShowLoads",b"ShowLoads",u"ShowStores",b"ShowStores",u"ShowTotals",b"ShowTotals"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowAtomics",b"ShowAtomics",u"ShowLoads",b"ShowLoads",u"ShowStores",b"ShowStores",u"ShowTotals",b"ShowTotals"]) -> None: ...
type___ProfilerSectionMemorySharedTable = ProfilerSectionMemorySharedTable

class ProfilerSectionMemoryFirstLevelCacheTable(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...
    ShowLoads: builtin___bool = ...
    ShowStores: builtin___bool = ...
    ShowAtomics: builtin___bool = ...
    ShowReductions: builtin___bool = ...
    ShowGlobal: builtin___bool = ...
    ShowLocal: builtin___bool = ...
    ShowSurface: builtin___bool = ...
    ShowTexture: builtin___bool = ...
    ShowTotalLoads: builtin___bool = ...
    ShowTotalStores: builtin___bool = ...
    ShowTotals: builtin___bool = ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        ShowLoads : typing___Optional[builtin___bool] = None,
        ShowStores : typing___Optional[builtin___bool] = None,
        ShowAtomics : typing___Optional[builtin___bool] = None,
        ShowReductions : typing___Optional[builtin___bool] = None,
        ShowGlobal : typing___Optional[builtin___bool] = None,
        ShowLocal : typing___Optional[builtin___bool] = None,
        ShowSurface : typing___Optional[builtin___bool] = None,
        ShowTexture : typing___Optional[builtin___bool] = None,
        ShowTotalLoads : typing___Optional[builtin___bool] = None,
        ShowTotalStores : typing___Optional[builtin___bool] = None,
        ShowTotals : typing___Optional[builtin___bool] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowAtomics",b"ShowAtomics",u"ShowGlobal",b"ShowGlobal",u"ShowLoads",b"ShowLoads",u"ShowLocal",b"ShowLocal",u"ShowReductions",b"ShowReductions",u"ShowStores",b"ShowStores",u"ShowSurface",b"ShowSurface",u"ShowTexture",b"ShowTexture",u"ShowTotalLoads",b"ShowTotalLoads",u"ShowTotalStores",b"ShowTotalStores",u"ShowTotals",b"ShowTotals"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowAtomics",b"ShowAtomics",u"ShowGlobal",b"ShowGlobal",u"ShowLoads",b"ShowLoads",u"ShowLocal",b"ShowLocal",u"ShowReductions",b"ShowReductions",u"ShowStores",b"ShowStores",u"ShowSurface",b"ShowSurface",u"ShowTexture",b"ShowTexture",u"ShowTotalLoads",b"ShowTotalLoads",u"ShowTotalStores",b"ShowTotalStores",u"ShowTotals",b"ShowTotals"]) -> None: ...
type___ProfilerSectionMemoryFirstLevelCacheTable = ProfilerSectionMemoryFirstLevelCacheTable

class ProfilerSectionMemorySecondLevelCacheTable(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...
    ShowLoads: builtin___bool = ...
    ShowStores: builtin___bool = ...
    ShowAtomics: builtin___bool = ...
    ShowReductions: builtin___bool = ...
    ShowGlobal: builtin___bool = ...
    ShowLocal: builtin___bool = ...
    ShowSurface: builtin___bool = ...
    ShowTexture: builtin___bool = ...
    ShowTotalLoads: builtin___bool = ...
    ShowTotalStores: builtin___bool = ...
    ShowTotals: builtin___bool = ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        ShowLoads : typing___Optional[builtin___bool] = None,
        ShowStores : typing___Optional[builtin___bool] = None,
        ShowAtomics : typing___Optional[builtin___bool] = None,
        ShowReductions : typing___Optional[builtin___bool] = None,
        ShowGlobal : typing___Optional[builtin___bool] = None,
        ShowLocal : typing___Optional[builtin___bool] = None,
        ShowSurface : typing___Optional[builtin___bool] = None,
        ShowTexture : typing___Optional[builtin___bool] = None,
        ShowTotalLoads : typing___Optional[builtin___bool] = None,
        ShowTotalStores : typing___Optional[builtin___bool] = None,
        ShowTotals : typing___Optional[builtin___bool] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowAtomics",b"ShowAtomics",u"ShowGlobal",b"ShowGlobal",u"ShowLoads",b"ShowLoads",u"ShowLocal",b"ShowLocal",u"ShowReductions",b"ShowReductions",u"ShowStores",b"ShowStores",u"ShowSurface",b"ShowSurface",u"ShowTexture",b"ShowTexture",u"ShowTotalLoads",b"ShowTotalLoads",u"ShowTotalStores",b"ShowTotalStores",u"ShowTotals",b"ShowTotals"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowAtomics",b"ShowAtomics",u"ShowGlobal",b"ShowGlobal",u"ShowLoads",b"ShowLoads",u"ShowLocal",b"ShowLocal",u"ShowReductions",b"ShowReductions",u"ShowStores",b"ShowStores",u"ShowSurface",b"ShowSurface",u"ShowTexture",b"ShowTexture",u"ShowTotalLoads",b"ShowTotalLoads",u"ShowTotalStores",b"ShowTotalStores",u"ShowTotals",b"ShowTotals"]) -> None: ...
type___ProfilerSectionMemorySecondLevelCacheTable = ProfilerSectionMemorySecondLevelCacheTable

class ProfilerSectionMemoryDeviceMemoryTable(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...
    ShowLoads: builtin___bool = ...
    ShowStores: builtin___bool = ...
    ShowTotals: builtin___bool = ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        ShowLoads : typing___Optional[builtin___bool] = None,
        ShowStores : typing___Optional[builtin___bool] = None,
        ShowTotals : typing___Optional[builtin___bool] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowLoads",b"ShowLoads",u"ShowStores",b"ShowStores",u"ShowTotals",b"ShowTotals"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"ShowLoads",b"ShowLoads",u"ShowStores",b"ShowStores",u"ShowTotals",b"ShowTotals"]) -> None: ...
type___ProfilerSectionMemoryDeviceMemoryTable = ProfilerSectionMemoryDeviceMemoryTable

class ProfilerSectionMemoryChart(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Label: typing___Text = ...

    def __init__(self,
        *,
        Label : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label"]) -> None: ...
type___ProfilerSectionMemoryChart = ProfilerSectionMemoryChart

class ProfilerSectionGfxMetricsWidget(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Type: typing___Text = ...
    Label: typing___Text = ...

    @property
    def Metrics(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetric]: ...

    def __init__(self,
        *,
        Type : typing___Optional[typing___Text] = None,
        Label : typing___Optional[typing___Text] = None,
        Metrics : typing___Optional[typing___Iterable[type___ProfilerSectionMetric]] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"Type",b"Type"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Label",b"Label",u"Metrics",b"Metrics",u"Type",b"Type"]) -> None: ...
type___ProfilerSectionGfxMetricsWidget = ProfilerSectionGfxMetricsWidget

class ProfilerSectionHeader(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Rows: builtin___int = ...

    @property
    def Metrics(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetric]: ...

    def __init__(self,
        *,
        Rows : typing___Optional[builtin___int] = None,
        Metrics : typing___Optional[typing___Iterable[type___ProfilerSectionMetric]] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Rows",b"Rows"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Metrics",b"Metrics",u"Rows",b"Rows"]) -> None: ...
type___ProfilerSectionHeader = ProfilerSectionHeader

class ProfilerSectionBodyItem(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def Table(self) -> type___ProfilerSectionTable: ...

    @property
    def BarChart(self) -> type___ProfilerSectionBarChart: ...

    @property
    def HistogramChart(self) -> type___ProfilerSectionHistogramChart: ...

    @property
    def LineChart(self) -> type___ProfilerSectionLineChart: ...

    @property
    def MemorySharedTable(self) -> type___ProfilerSectionMemorySharedTable: ...

    @property
    def MemoryFirstLevelCacheTable(self) -> type___ProfilerSectionMemoryFirstLevelCacheTable: ...

    @property
    def MemorySecondLevelCacheTable(self) -> type___ProfilerSectionMemorySecondLevelCacheTable: ...

    @property
    def MemoryDeviceMemoryTable(self) -> type___ProfilerSectionMemoryDeviceMemoryTable: ...

    @property
    def MemoryChart(self) -> type___ProfilerSectionMemoryChart: ...

    @property
    def GfxMetricsWidget(self) -> type___ProfilerSectionGfxMetricsWidget: ...

    @property
    def Filter(self) -> ProfilerMetricOptions_pb2___MetricOptionFilter: ...

    def __init__(self,
        *,
        Table : typing___Optional[type___ProfilerSectionTable] = None,
        BarChart : typing___Optional[type___ProfilerSectionBarChart] = None,
        HistogramChart : typing___Optional[type___ProfilerSectionHistogramChart] = None,
        LineChart : typing___Optional[type___ProfilerSectionLineChart] = None,
        MemorySharedTable : typing___Optional[type___ProfilerSectionMemorySharedTable] = None,
        MemoryFirstLevelCacheTable : typing___Optional[type___ProfilerSectionMemoryFirstLevelCacheTable] = None,
        MemorySecondLevelCacheTable : typing___Optional[type___ProfilerSectionMemorySecondLevelCacheTable] = None,
        MemoryDeviceMemoryTable : typing___Optional[type___ProfilerSectionMemoryDeviceMemoryTable] = None,
        MemoryChart : typing___Optional[type___ProfilerSectionMemoryChart] = None,
        GfxMetricsWidget : typing___Optional[type___ProfilerSectionGfxMetricsWidget] = None,
        Filter : typing___Optional[ProfilerMetricOptions_pb2___MetricOptionFilter] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"BarChart",b"BarChart",u"Filter",b"Filter",u"GfxMetricsWidget",b"GfxMetricsWidget",u"HistogramChart",b"HistogramChart",u"LineChart",b"LineChart",u"MemoryChart",b"MemoryChart",u"MemoryDeviceMemoryTable",b"MemoryDeviceMemoryTable",u"MemoryFirstLevelCacheTable",b"MemoryFirstLevelCacheTable",u"MemorySecondLevelCacheTable",b"MemorySecondLevelCacheTable",u"MemorySharedTable",b"MemorySharedTable",u"Table",b"Table"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"BarChart",b"BarChart",u"Filter",b"Filter",u"GfxMetricsWidget",b"GfxMetricsWidget",u"HistogramChart",b"HistogramChart",u"LineChart",b"LineChart",u"MemoryChart",b"MemoryChart",u"MemoryDeviceMemoryTable",b"MemoryDeviceMemoryTable",u"MemoryFirstLevelCacheTable",b"MemoryFirstLevelCacheTable",u"MemorySecondLevelCacheTable",b"MemorySecondLevelCacheTable",u"MemorySharedTable",b"MemorySharedTable",u"Table",b"Table"]) -> None: ...
type___ProfilerSectionBodyItem = ProfilerSectionBodyItem

class ProfilerSectionBody(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    DisplayName: typing___Text = ...

    @property
    def Items(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionBodyItem]: ...

    def __init__(self,
        *,
        Items : typing___Optional[typing___Iterable[type___ProfilerSectionBodyItem]] = None,
        DisplayName : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"DisplayName",b"DisplayName"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"DisplayName",b"DisplayName",u"Items",b"Items"]) -> None: ...
type___ProfilerSectionBody = ProfilerSectionBody

class ProfilerSectionMetrics(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Order: builtin___int = ...

    @property
    def Metrics(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionMetric]: ...

    def __init__(self,
        *,
        Metrics : typing___Optional[typing___Iterable[type___ProfilerSectionMetric]] = None,
        Order : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Order",b"Order"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Metrics",b"Metrics",u"Order",b"Order"]) -> None: ...
type___ProfilerSectionMetrics = ProfilerSectionMetrics

class ProfilerSection(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    Identifier: typing___Text = ...
    DisplayName: typing___Text = ...
    Order: builtin___int = ...
    Description: typing___Text = ...
    Extends: typing___Text = ...

    @property
    def Header(self) -> type___ProfilerSectionHeader: ...

    @property
    def Body(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSectionBody]: ...

    @property
    def Metrics(self) -> type___ProfilerSectionMetrics: ...

    def __init__(self,
        *,
        Identifier : typing___Optional[typing___Text] = None,
        DisplayName : typing___Optional[typing___Text] = None,
        Order : typing___Optional[builtin___int] = None,
        Header : typing___Optional[type___ProfilerSectionHeader] = None,
        Body : typing___Optional[typing___Iterable[type___ProfilerSectionBody]] = None,
        Metrics : typing___Optional[type___ProfilerSectionMetrics] = None,
        Description : typing___Optional[typing___Text] = None,
        Extends : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"Description",b"Description",u"DisplayName",b"DisplayName",u"Extends",b"Extends",u"Header",b"Header",u"Identifier",b"Identifier",u"Metrics",b"Metrics",u"Order",b"Order"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Body",b"Body",u"Description",b"Description",u"DisplayName",b"DisplayName",u"Extends",b"Extends",u"Header",b"Header",u"Identifier",b"Identifier",u"Metrics",b"Metrics",u"Order",b"Order"]) -> None: ...
type___ProfilerSection = ProfilerSection

class ProfilerSections(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def Sections(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ProfilerSection]: ...

    def __init__(self,
        *,
        Sections : typing___Optional[typing___Iterable[type___ProfilerSection]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"Sections",b"Sections"]) -> None: ...
type___ProfilerSections = ProfilerSections
