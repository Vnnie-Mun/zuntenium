"""transpiler sub-package."""
from __future__ import annotations

from zuntenium.transpiler.coupling_map import CouplingMap
from zuntenium.transpiler.pipeline import TranspilerPipeline
from zuntenium.transpiler.target import Target
from zuntenium.transpiler.cache import TranspilerCache, CachedCompiledTemplate

# Lazy import to avoid pulling networkx/numpy at package-init time
# when user only needs the pipeline.


def __getattr__(name: str) -> object:
    if name == "AITranspiler":
        from zuntenium.transpiler.ai.transpiler import AITranspiler

        return AITranspiler
    if name == "transpiler_v2":
        from zuntenium.transpiler import transpiler_v2
        return transpiler_v2
    if name in {
        "HardwareConfig", "HardwareProfileRegistry", "HardwareCostFunction",
        "QubitMapping", "QubitMappingPass", "SWAPInsertionPass",
        "HardwareCostOptPass", "HardwareRotationMergingPass", "GateDecomposer", "GateDecompositionPass",
        "HardwareTranspilerResult", "HardwareAwareTranspiler", "build_hardware_executor",
    }:
        from zuntenium.transpiler import hardware_transpiler

        return getattr(hardware_transpiler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TranspilerPipeline", "CouplingMap", "Target", "AITranspiler",
    "transpiler_v2", "TranspilerCache", "CachedCompiledTemplate",
    "HardwareConfig", "HardwareProfileRegistry", "HardwareCostFunction",
    "QubitMapping", "QubitMappingPass", "SWAPInsertionPass",
    "HardwareCostOptPass", "HardwareRotationMergingPass", "GateDecomposer", "GateDecompositionPass",
    "HardwareTranspilerResult", "HardwareAwareTranspiler", "build_hardware_executor",
]

