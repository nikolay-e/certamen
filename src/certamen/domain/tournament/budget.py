import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from certamen.shared.logging.structured import ContextualLogger

from certamen.shared.logging import get_contextual_logger


class CostTracker:
    def __init__(
        self, logger: "logging.Logger | ContextualLogger | None" = None
    ):
        self.total_cost = 0.0
        self.cost_by_model: dict[str, float] = {}
        self.logger = logger or get_contextual_logger("certamen.cost_tracker")

    def add_cost(self, model_name: str, cost: float) -> None:
        self.total_cost += cost

        if model_name not in self.cost_by_model:
            self.cost_by_model[model_name] = 0.0
        self.cost_by_model[model_name] += cost

        self.logger.debug(
            "Added $%.4f for %s, total now: $%.4f",
            cost,
            model_name,
            self.total_cost,
        )

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_cost": f"${self.total_cost:.4f}",
            "cost_by_model": {
                k: f"${v:.4f}" for k, v in self.cost_by_model.items()
            },
        }

    def display_summary(self) -> None:
        self.logger.info(
            "Cost Summary", extra={"display_type": "section_header"}
        )
        self.logger.info(
            "Total Cost: $%.4f",
            self.total_cost,
            extra={"display_type": "colored_text"},
        )

        if self.cost_by_model:
            self.logger.info(
                "\nCost by Model:", extra={"display_type": "colored_text"}
            )
            for model_name, cost in sorted(
                self.cost_by_model.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (
                    (cost / self.total_cost) * 100
                    if self.total_cost > 0
                    else 0
                )
                self.logger.info(
                    "  %s: $%.4f (%.1f%%)",
                    model_name,
                    cost,
                    percentage,
                    extra={"display_type": "colored_text"},
                )
        else:
            self.logger.info(
                "No cost information available (cost tracking may be disabled)",
                extra={"display_type": "colored_text"},
            )

        self.logger.info("Tournament total cost: $%.4f", self.total_cost)
