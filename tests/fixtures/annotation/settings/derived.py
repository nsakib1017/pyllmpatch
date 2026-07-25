# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: '/home/jjf190004/pyllm_dataset/fastloom/84cf4216143a3347287fd2d51db2c69471ec2cae8282ae9ad24cc35d1bab8de85f050bae6849a0ee4dd61d32adf629365ade0f45c518570aa3f494640585bc61/settings.py'
# Bytecode version: 3.15 (3666)
# Source timestamp: 2026-07-08 16:38:45 UTC (1783528725)

from typing import Annotated, Literal
from pydantic import AnyHttpUrl, BaseModel, BeforeValidator, Field
from fastloom.settings.base import MonitoringSettings
from fastloom.settings.utils import pydantic_env_or_default
from fastloom.types import Str
type ExporterType = Literal['otlp', 'console', 'none']
type MetricsExporterType = ExporterType | Literal['prometheus']
type TracesExporterType = ExporterType | Literal['zipkin']
type EnvBackend[T] = Annotated[T, BeforeValidator(pydantic_env_or_default)]
def EnvDefault[T](default: T):
    return Field(default=default, validate_default=True)
class OtelConfig(BaseModel):
    # ***<module>.OtelConfig: Failure: Different bytecode
    OTEL_EXPORTER_OTLP_ENDPOINT = EnvDefault(None)
    OTEL_EXPORTER_OTLP_INSECURE = EnvDefault(True)
    OTEL_EXPORTER_OTLP_PROTOCOL = EnvDefault('http/protobuf')
    OTEL_LOGS_EXPORTER = EnvDefault('otlp')
    OTEL_METRICS_EXPORTER = EnvDefault('otlp')
    OTEL_TRACES_EXPORTER = EnvDefault('otlp')
    OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST = EnvDefault('.*')
    OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_RESPONSE = EnvDefault('.*')
    OTEL_EXPORTER_OTLP_ENDPOINT: EnvBackend[Str[AnyHttpUrl]] | None
class ObservabilitySettings(MonitoringSettings, OtelConfig):
    # ***<module>.ObservabilitySettings: Failure: Different bytecode
    SENTRY_ENABLED = 0
    OTEL_ENABLED = 0
    SENTRY_DSN = None
    METRICS = False
    SENTRY_ENABLED: int