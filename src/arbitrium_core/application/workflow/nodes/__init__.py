from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
)


def register_all() -> None:
    import arbitrium_core.application.workflow.nodes.evaluation
    import arbitrium_core.application.workflow.nodes.flow
    import arbitrium_core.application.workflow.nodes.generation
    import arbitrium_core.application.workflow.nodes.input
    import arbitrium_core.application.workflow.nodes.knowledge
    import arbitrium_core.application.workflow.nodes.llm
    import arbitrium_core.application.workflow.nodes.output

    del (
        arbitrium_core.application.workflow.nodes.evaluation,
        arbitrium_core.application.workflow.nodes.flow,
        arbitrium_core.application.workflow.nodes.generation,
        arbitrium_core.application.workflow.nodes.input,
        arbitrium_core.application.workflow.nodes.knowledge,
        arbitrium_core.application.workflow.nodes.llm,
        arbitrium_core.application.workflow.nodes.output,
    )


__all__ = ["BaseNode", "ExecutionContext", "Port", "PortType", "register_all"]
