"""Public API contract for the reusable VClasp core."""

import vclasp


PRODUCTION_EXECUTORS = (
    vclasp.LocalVClaspExecutor,
    vclasp.S3VClaspExecutor,
    vclasp.AIStoreVClaspExecutor,
    vclasp.S3VClaspExecutorPool,
)


def test_production_executors_share_one_request_interface():
    for executor in PRODUCTION_EXECUTORS:
        methods = set(dir(executor))
        assert {"execute", "execute_window"} <= methods
        assert "execute_forced" not in methods
        assert "execute_partitioned" not in methods
        assert "execute_on_worker" not in methods
        assert "plan_dependency_sampler" not in methods
        assert "plan_label_preserving_dependency_sampler" not in methods
        assert "plan_label_preserving_logical_sampler" not in methods
        assert "choose_lookahead" not in methods
        assert "inspect_sparse_candidates" not in methods


def test_historical_readers_are_not_public_module_members():
    historical = {
        "PyLogicalBatchDecoder",
        "PyLogicalScheduler",
        "PyAdaptiveBatchExecutor",
        "PyNormalizedBatchExecutor",
        "PyPairBatchExecutor",
        "PyPrefixBatchExecutor",
    }
    assert historical.isdisjoint(dir(vclasp))
