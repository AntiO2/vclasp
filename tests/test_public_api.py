"""Public API contract for the reusable VClasp core."""

import vclasp


PRODUCTION_EXECUTORS = (vclasp.VClaspSession,)


def test_production_executors_share_one_request_interface():
    for executor in PRODUCTION_EXECUTORS:
        methods = set(dir(executor))
        assert {
            "execute",
            "execute_window",
            "submit",
            "submit_window",
            "pipeline",
            "metrics_snapshot",
            "synchronize",
        } <= methods
        assert "execute_forced" not in methods
        assert "execute_partitioned" not in methods
        assert "execute_on_worker" not in methods
        assert "execute_session" not in methods
        assert "plan_dependency_sampler" not in methods
        assert "plan_label_preserving_dependency_sampler" not in methods
        assert "plan_label_preserving_logical_sampler" not in methods
        assert "choose_lookahead" not in methods
        assert "inspect_sparse_candidates" not in methods


def test_public_entry_points_are_explicit():
    assert callable(vclasp.build_vclasp_chunk)
    assert callable(vclasp.plan_byte_ranges)
    assert callable(vclasp.VClaspSession.local)
    assert callable(vclasp.VClaspSession.aistore)


def test_historical_readers_are_not_public_module_members():
    historical = {
        "LocalVClaspExecutor",
        "S3VClaspExecutor",
        "AIStoreVClaspExecutor",
        "PyLogicalBatchDecoder",
        "PyLogicalScheduler",
        "PyAdaptiveBatchExecutor",
        "PyNormalizedBatchExecutor",
        "PyPairBatchExecutor",
        "PyPrefixBatchExecutor",
        "H264Decoder",
        "ByteCache",
        "S3RangeReader",
        "S3ObjectStoreReader",
        "LocalRangeReader",
        "AIStoreGetBatchReader",
    }
    assert historical.isdisjoint(dir(vclasp))


def test_legacy_session_aliases_are_not_public():
    assert "S3VClaspSession" not in dir(vclasp)
    assert "S3VClaspExecutorPool" not in dir(vclasp)


def test_pipeline_handles_are_first_class_public_results():
    assert {"result"} <= set(dir(vclasp.PendingBatch))
    assert {"result"} <= set(dir(vclasp.PendingWindow))
    assert {
        "submit",
        "take",
        "close",
        "is_closed",
        "metrics_snapshot",
    } <= set(dir(vclasp.VClaspPipeline))


def test_chunk_inspection_does_not_expose_an_alternative_decoder():
    direct_execution = {
        "decode_hierarchical_target_rgb24",
        "decode_record_rgb24",
        "decode_records_batch_rgb24",
        "decode_gop_record_rgb24",
        "decode_gop_records_batch_rgb24",
        "decode_gop_all_records_rgb24",
        "decode_gop_record_at",
        "create_logical_scheduler",
    }
    assert direct_execution.isdisjoint(dir(vclasp.VClaspChunk))
