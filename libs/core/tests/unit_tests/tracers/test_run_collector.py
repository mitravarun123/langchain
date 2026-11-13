"""Test the run collector."""

import uuid
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from langchain_core.language_models import FakeListLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.schemas import Run


def _make_run(
    *,
    name: str = "run",
    run_type: str = "chain",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    error: str | None = None,
    parent_run_id: uuid.UUID | None = None,
) -> Run:
    if start_time is None:
        start_time = datetime(2024, 1, 1, tzinfo=UTC)
    return Run(
        id=uuid4(),
        name=name,
        run_type=run_type,
        start_time=start_time,
        end_time=end_time,
        error=error,
        parent_run_id=parent_run_id,
        inputs={"prompt": "private prompt"},
        outputs={"response": "private response"},
        extra={"metadata": {"secret": "private metadata"}},
    )


def test_collect_runs() -> None:
    model = FakeListLLM(responses=["hello"])
    with collect_runs() as cb:
        model.invoke("hi")
        assert cb.traced_runs
        assert len(cb.traced_runs) == 1
        assert isinstance(cb.traced_runs[0].id, uuid.UUID)
        assert cb.traced_runs[0].inputs == {"prompts": ["hi"]}
        summaries = cb.get_run_summaries()
        assert len(summaries) == 1
        assert summaries[0]["status"] == "success"


def test_get_run_summaries_empty_collector() -> None:
    collector = RunCollectorCallbackHandler()

    assert collector.get_run_summaries() == []


def test_get_run_summaries_uses_timestamps_and_excludes_content() -> None:
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    run = _make_run(
        name="private-name",
        start_time=start_time,
        end_time=start_time + timedelta(milliseconds=1250),
    )
    collector = RunCollectorCallbackHandler()
    collector.traced_runs = [run]

    summary = collector.get_run_summaries()[0]

    assert summary == {
        "run_id": str(run.id),
        "parent_run_id": None,
        "name": "private-name",
        "run_type": "chain",
        "duration_ms": 1250.0,
        "status": "success",
    }
    assert "private prompt" not in str(summary)
    assert "private response" not in str(summary)
    assert "private metadata" not in str(summary)
    assert "error" not in summary


def test_get_run_summaries_flattens_multiple_nested_roots_in_order() -> None:
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    first_root = _make_run(
        name="agent",
        start_time=start_time,
        end_time=start_time + timedelta(seconds=3),
    )
    llm_run = _make_run(
        name="model",
        run_type="llm",
        start_time=start_time + timedelta(seconds=1),
        end_time=start_time + timedelta(seconds=2),
        parent_run_id=first_root.id,
    )
    tool_run = _make_run(
        name="tool",
        run_type="tool",
        start_time=start_time + timedelta(milliseconds=1500),
        end_time=start_time + timedelta(milliseconds=2500),
    )
    first_root.child_runs = [llm_run, tool_run]
    second_root = _make_run(
        name="second-agent",
        start_time=start_time,
        end_time=start_time + timedelta(milliseconds=500),
    )
    collector = RunCollectorCallbackHandler()
    collector.traced_runs = [first_root, second_root]
    original_root_children = list(first_root.child_runs)
    original_run_count = len(collector.traced_runs)
    original_root_inputs = dict(first_root.inputs)
    original_child_ids = [child.id for child in first_root.child_runs]

    summaries = collector.get_run_summaries()

    assert [summary["name"] for summary in summaries] == [
        "agent",
        "model",
        "tool",
        "second-agent",
    ]
    assert [summary["parent_run_id"] for summary in summaries] == [
        None,
        str(first_root.id),
        str(first_root.id),
        None,
    ]
    assert [summary["duration_ms"] for summary in summaries] == [
        3000.0,
        1000.0,
        1000.0,
        500.0,
    ]
    assert first_root.child_runs == original_root_children
    assert first_root.inputs == original_root_inputs
    assert [child.id for child in first_root.child_runs] == original_child_ids
    assert len(collector.traced_runs) == original_run_count


@pytest.mark.parametrize(
    ("start_time", "end_time", "error", "duration_ms", "status"),
    [
        (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
            "private error text",
            1000.0,
            "error",
        ),
        (datetime(2024, 1, 1, tzinfo=UTC), None, None, None, "incomplete"),
        (None, datetime(2024, 1, 1, tzinfo=UTC), None, None, "success"),
    ],
)
def test_get_run_summaries_status_and_incomplete_timestamps(
    start_time: datetime | None,
    end_time: datetime | None,
    error: str | None,
    duration_ms: float | None,
    status: str,
) -> None:
    run = _make_run(
        start_time=start_time,
        end_time=end_time,
        error=error,
    )
    if start_time is None:
        object.__setattr__(run, "start_time", None)
    collector = RunCollectorCallbackHandler()
    collector.traced_runs = [run]

    summary = collector.get_run_summaries()[0]

    assert summary["duration_ms"] == duration_ms
    assert summary["status"] == status
    assert "private error text" not in str(summary)


def test_get_run_summaries_handles_deep_trees_iteratively() -> None:
    child: Run | None = None
    depth = 1500
    for index in range(depth):
        run = _make_run(name=f"run-{index}")
        if child is not None:
            run.child_runs = [child]
        child = run
    assert child is not None
    collector = RunCollectorCallbackHandler()
    collector.traced_runs = [child]

    summaries = collector.get_run_summaries()

    assert len(summaries) == depth
    assert summaries[0]["name"] == "run-1499"
    assert summaries[-1]["name"] == "run-0"


def test_get_run_summaries_for_failing_runnable() -> None:
    def fail(_: str) -> str:
        msg = "private tool error"
        raise ValueError(msg)

    runnable = RunnableLambda(fail)
    with collect_runs() as collector, pytest.raises(ValueError, match="private tool"):
        runnable.invoke("private input")

    summary = collector.get_run_summaries()[0]
    assert summary["status"] == "error"
    assert "private tool error" not in str(summary)
    assert "private input" not in str(summary)


@pytest.mark.asyncio
async def test_get_run_summaries_for_async_runnable() -> None:
    async def double(value: int) -> int:
        return value * 2

    runnable = RunnableLambda(lambda value: value * 2, afunc=double)
    with collect_runs() as collector:
        assert await runnable.ainvoke(3) == 6

    summaries = collector.get_run_summaries()
    assert len(summaries) == 1
    assert summaries[0]["status"] == "success"
    assert summaries[0]["run_type"] == "chain"
