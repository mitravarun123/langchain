"""A tracer that collects all nested runs in a list."""

from typing import Any, Literal, TypedDict
from uuid import UUID

from langchain_core.tracers._compat import run_copy
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class RunSummary(TypedDict):
    """Content-free summary of a collected run."""

    run_id: str
    parent_run_id: str | None
    name: str
    run_type: str
    duration_ms: float | None
    status: Literal["success", "error", "incomplete"]


class RunCollectorCallbackHandler(BaseTracer):
    """Tracer that collects all nested runs in a list.

    This tracer is useful for inspection and evaluation purposes.
    """

    name: str = "run-collector_callback_handler"

    def __init__(self, example_id: UUID | str | None = None, **kwargs: Any) -> None:
        """Initialize the `RunCollectorCallbackHandler`.

        Args:
            example_id: The ID of the example being traced.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.traced_runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run by adding it to the `traced_runs` list.

        Args:
            run: The run to be persisted.
        """
        run_ = run_copy(run)
        run_.reference_example_id = self.example_id
        self.traced_runs.append(run_)

    def get_run_summaries(self) -> list[RunSummary]:
        """Return content-free summaries of the collected runs.

        Runs are returned in root insertion order and depth-first child order.
        Durations are inclusive wall-clock durations calculated from recorded
        timestamps; nested durations are not subtracted from their parents. A
        run with an error is marked `error`, one without an end time is marked
        `incomplete`, and other ended runs are marked `success`.

        !!! warning "Beta"
            This local summary is experimental and only reflects runs already
            present in this collector. It does not provide live progress or
            replace LangSmith tracing.

        Returns:
            A list of summaries containing IDs, names, run types, durations,
            and statuses. Inputs, outputs, metadata, and error messages are
            excluded. Run names remain in the summary, so applications should
            avoid putting sensitive information in run names.

        Example:
            >>> from langchain_core.runnables import RunnableLambda
            >>> from langchain_core.tracers.context import collect_runs
            >>> runnable = RunnableLambda(lambda text: text.upper())
            >>> with collect_runs() as collector:
            ...     runnable.invoke("hello")
            >>> collector.get_run_summaries()[0]["status"]
            'success'
        """
        summaries: list[RunSummary] = []
        pending: list[tuple[Run, str | None]] = [
            (run, None) for run in reversed(self.traced_runs)
        ]

        while pending:
            run, traversal_parent_id = pending.pop()
            parent_run_id = run.parent_run_id or traversal_parent_id
            start_time = run.start_time
            end_time = run.end_time
            duration_ms = (
                (end_time - start_time).total_seconds() * 1000
                if start_time is not None and end_time is not None
                else None
            )
            status: Literal["success", "error", "incomplete"]
            if run.error is not None:
                status = "error"
            elif end_time is None:
                status = "incomplete"
            else:
                status = "success"

            summaries.append(
                {
                    "run_id": str(run.id),
                    "parent_run_id": (
                        str(parent_run_id) if parent_run_id is not None else None
                    ),
                    "name": run.name,
                    "run_type": run.run_type,
                    "duration_ms": duration_ms,
                    "status": status,
                }
            )
            pending.extend((child, str(run.id)) for child in reversed(run.child_runs))

        return summaries
