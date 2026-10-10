"""AI optimization and execution-history components."""

from zuntenium.ai_engine.ai_optimizer_part1 import (
    BACKEND_LABELS,
    TRANSPILE_LABELS,
    ExecutionLogger,
    ExecutionRecord,
    FeatureEngineer,
    FeatureVector,
    HistoricalContext,
    RuntimeContext,
    compute_reward,
    Part1TestSuite,
)
from zuntenium.ai_engine.ai_optimizer_part2 import (
    AIDecision,
    AIOptimizer,
    RLAgent,
    SupervisedModel,
    action_to_bs,
    bs_to_action,
    make_ai_hybrid_executor,
    Part2TestSuite,
    run_demo,
)

__all__ = [
    "BACKEND_LABELS", "TRANSPILE_LABELS", "ExecutionLogger", "ExecutionRecord",
    "FeatureEngineer", "FeatureVector", "HistoricalContext", "RuntimeContext",
    "compute_reward", "Part1TestSuite",
    "AIDecision", "AIOptimizer", "RLAgent", "SupervisedModel",
    "action_to_bs", "bs_to_action", "make_ai_hybrid_executor",
    "Part2TestSuite", "run_demo",
    "EngineFeatureVector", "FeatureExtractor", "EngineExecutionLogger",
    "AsyncExecutionLogger",
    "EngineAIDecision", "AdaptiveAIOptimizer", "EngineSupervisedModel",
    "PPOAgent", "ReplayBuffer", "OnlineLearningLoop",
]


_EXTENDED_EXPORTS = {
    "EngineFeatureVector": ("features", "FeatureVector"),
    "FeatureExtractor": ("features", "FeatureExtractor"),
    "EngineExecutionLogger": ("logger", "ExecutionLogger"),
    "AsyncExecutionLogger": ("logger", "AsyncExecutionLogger"),
    "EngineAIDecision": ("decision", "AIDecision"),
    "AdaptiveAIOptimizer": ("optimizer", "AIOptimizer"),
    "EngineSupervisedModel": ("models", "SupervisedModel"),
    "PPOAgent": ("models", "PPOAgent"),
    "ReplayBuffer": ("models", "ReplayBuffer"),
    "OnlineLearningLoop": ("online_loop", "OnlineLearningLoop"),
}


def __getattr__(name: str):
    if name not in _EXTENDED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute_name = _EXTENDED_EXPORTS[name]
    attribute = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = attribute
    return attribute
