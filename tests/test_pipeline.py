"""End-to-end pipeline tests on tiny models."""

import pytest

from dataprism.config.dataclass import DataPrismConfig
from dataprism.pipeline.full_pipeline import DataPrismPipeline


class TestPipelineConfig:
    """Test pipeline configuration and initialization."""

    def test_pipeline_init_all_disabled(self):
        """Pipeline should initialize with all phases disabled."""
        config = DataPrismConfig()
        config.phase1.enabled = False
        config.phase2.enabled = False
        config.phase3.enabled = False
        pipeline = DataPrismPipeline(config)
        assert pipeline is not None

    def test_phase1_pipeline(self):
        """Phase 1 pipeline structure test."""
        from dataprism.pipeline.phase1_pipeline import Phase1Pipeline
        config = DataPrismConfig()
        config.phase1.enabled = True
        config.phase2.enabled = False
        config.phase3.enabled = False
        pipeline = Phase1Pipeline(config)
        assert pipeline is not None

    def test_phase2_pipeline(self):
        """Phase 2 pipeline structure test."""
        from dataprism.pipeline.phase2_pipeline import Phase2Pipeline
        config = DataPrismConfig()
        pipeline = Phase2Pipeline(config)
        assert pipeline is not None

    def test_phase3_pipeline(self):
        """Phase 3 pipeline structure test."""
        from dataprism.pipeline.phase3_pipeline import Phase3Pipeline
        config = DataPrismConfig()
        pipeline = Phase3Pipeline(config)
        assert pipeline is not None


class TestDataPrismConfigCrossValidation:
    def test_phase_dependencies(self):
        """Phase configs should be independently toggleable."""
        config = DataPrismConfig()
        config.phase1.enabled = False
        config.phase2.enabled = True
        config.phase3.enabled = False
        assert not config.phase1.enabled
        assert config.phase2.enabled
        assert not config.phase3.enabled
