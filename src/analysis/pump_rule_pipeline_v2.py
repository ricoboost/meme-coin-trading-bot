"""Orchestrate Pump.fun V2 offline research pipeline."""

from __future__ import annotations

from src.analysis.pump_build_rule_packs_v2 import build_rule_packs_v2
from src.analysis.pump_mine_rules_v2 import mine_pump_rules_v2
from src.analysis.pump_prepare_dataset_v2 import build_pump_research_datasets
from src.utils.logging_utils import configure_logging


def main() -> None:
    """Run dataset preparation then V2 rule mining."""
    logger = configure_logging(logger_name=__name__)
    logger.info("Starting Pump V2 dataset preparation")
    build_pump_research_datasets()
    logger.info("Starting Pump V2 rule mining")
    mine_pump_rules_v2()
    logger.info("Starting Pump V2 rule pack build")
    build_rule_packs_v2()
    logger.info("Completed Pump V2 offline rule pipeline")


if __name__ == "__main__":
    main()
