"""lcmsdeconv: neural-network deconvolution and quantitation for LC-ESI-MS of macromolecules."""

import os as _os

__version__ = "0.1.2"

# ONNX Runtime's telemetry off unless the user has set the variable (see
# lcmsdeconv.nn.infer.import_onnxruntime). Set when lcmsdeconv is imported, so that it also holds
# for programs that import lcmsdeconv before onnxruntime: the Linux build reads it on import.
_os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
