from certamen.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
)


def register_all() -> None:
    import certamen.application.workflow.nodes.disagreement
    import certamen.application.workflow.nodes.evaluation
    import certamen.application.workflow.nodes.flow
    import certamen.application.workflow.nodes.generation
    import certamen.application.workflow.nodes.input
    import certamen.application.workflow.nodes.interrogation
    import certamen.application.workflow.nodes.knowledge
    import certamen.application.workflow.nodes.llm
    import certamen.application.workflow.nodes.output

    del (
        certamen.application.workflow.nodes.disagreement,
        certamen.application.workflow.nodes.evaluation,
        certamen.application.workflow.nodes.flow,
        certamen.application.workflow.nodes.generation,
        certamen.application.workflow.nodes.input,
        certamen.application.workflow.nodes.interrogation,
        certamen.application.workflow.nodes.knowledge,
        certamen.application.workflow.nodes.llm,
        certamen.application.workflow.nodes.output,
    )


__all__ = ["BaseNode", "ExecutionContext", "Port", "PortType", "register_all"]
