import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gleipnir.campaign_status import CampaignStatus


def test_parallel_jobs_publish_failure_before_other_lane_finishes(tmp_path):
    path = tmp_path / "status.json"
    status = CampaignStatus(path, [{"job_name": n} for n in ("good", "bad")], "rev")
    ready = threading.Event()
    release = threading.Event()

    def good_lane():
        with status.job("good", gpu=0):
            ready.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(good_lane)
        try:
            assert ready.wait(5)
            with pytest.raises(RuntimeError, match="invalid metadata"):
                with status.job("bad", gpu=1):
                    raise RuntimeError("invalid metadata")
            snapshot = json.loads(path.read_text())
            assert snapshot["state"] == "failed"
            assert snapshot["active_jobs"] == ["good"]
            assert snapshot["failed_jobs"] == ["bad"]
            assert snapshot["completed_jobs"] == []
            assert snapshot["job_status"]["bad"]["gpu"] == 1
            assert "invalid metadata" in snapshot["job_status"]["bad"]["error"]
        finally:
            release.set()
        future.result()
    final = json.loads(path.read_text())
    assert final["state"] == "failed"
    assert final["active_jobs"] == []
    assert final["completed_jobs"] == ["good"]


def test_status_transitions_keep_observation_times_and_metadata(tmp_path):
    path = tmp_path / "nested" / "status.json"
    status = CampaignStatus(
        path, [{"job_name": "a"}], None, metadata={"campaign_id": "example"}
    )
    status.start("a", gpu=1)
    first = json.loads(path.read_text())
    status.start("a", gpu=1)
    assert json.loads(path.read_text())["job_status"] == first["job_status"]
    status.finish("a")
    status.finish("a")
    status.update(state="complete", phase="complete")
    final = json.loads(path.read_text())
    assert final["completed_jobs"] == ["a"]
    assert final["campaign_id"] == "example"
    assert final["updated_at_unix"] >= first["updated_at_unix"]
    with pytest.raises(ValueError, match="terminal"):
        status.start("a")
    with pytest.raises(ValueError, match="unknown"):
        status.start("missing")


def test_metadata_cannot_override_lifecycle(tmp_path):
    with pytest.raises(ValueError, match="lifecycle"):
        CampaignStatus(
            tmp_path / "status.json", [], None, metadata={"state": "complete"}
        )
