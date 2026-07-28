"""
Nested tracing for the single job-search agent pipeline.

Uses Langfuse when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set;
otherwise records a local nested span tree under one root trace so
inputs/outputs/config stay visible in-notebook.

When Langfuse is enabled, ALL spans nest under ONE root observation so the
dashboard shows a single end-to-end trace (not one fragment per tool call).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4
import json
import os


_CURRENT_SPAN: ContextVar["SpanRecord | None"] = ContextVar(
    "pipeline_current_span",
    default=None,
)
_ROOT_TRACE: ContextVar["TraceRecord | None"] = ContextVar(
    "pipeline_root_trace",
    default=None,
)

_LANGFUSE_CLIENT = None
_LANGFUSE_CHECKED = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except TypeError:
        return str(value)


@dataclass
class SpanRecord:
    name: str
    span_id: str = field(default_factory=lambda: uuid4().hex)
    parent_id: str | None = None
    start_time: str = field(default_factory=_utc_now)
    end_time: str | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list["SpanRecord"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "error": self.error,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class TraceRecord:
    name: str
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    start_time: str = field(default_factory=_utc_now)
    end_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    root_spans: list[SpanRecord] = field(default_factory=list)
    public_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
            "public_url": self.public_url,
            "spans": [span.to_dict() for span in self.root_spans],
        }


def langfuse_enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def reset_langfuse_client() -> None:
    """Force re-read of env keys (useful after load_dotenv)."""
    global _LANGFUSE_CLIENT, _LANGFUSE_CHECKED
    _LANGFUSE_CLIENT = None
    _LANGFUSE_CHECKED = False


def _get_langfuse():
    global _LANGFUSE_CLIENT, _LANGFUSE_CHECKED
    if _LANGFUSE_CHECKED:
        return _LANGFUSE_CLIENT
    _LANGFUSE_CHECKED = True
    if not langfuse_enabled():
        return None
    try:
        from langfuse import get_client

        _LANGFUSE_CLIENT = get_client()
        return _LANGFUSE_CLIENT
    except Exception as exc:  # noqa: BLE001 — optional dependency
        print(f"[tracing] Langfuse unavailable ({exc}); using local spans.")
        _LANGFUSE_CLIENT = None
        return None


def _start_langfuse_span(
    langfuse: Any,
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    as_type: str = "span",
):
    """Prefer start_as_current_span; fall back to start_as_current_observation."""
    kwargs: dict[str, Any] = {
        "name": name,
        "metadata": dict(metadata or {}),
    }
    if input is not None:
        kwargs["input"] = _jsonable(input)

    if hasattr(langfuse, "start_as_current_span") and as_type == "span":
        return langfuse.start_as_current_span(**kwargs)

    return langfuse.start_as_current_observation(as_type=as_type, **kwargs)


def _capture_langfuse_url(langfuse: Any, trace: TraceRecord) -> None:
    """
    Capture public URL / Langfuse trace id WHILE a span is still active.

    get_trace_url() returns None after the root context exits, so this must
    run before __exit__ on the root observation.
    """
    tid = None
    url = None
    try:
        tid_fn = getattr(langfuse, "get_current_trace_id", None)
        if callable(tid_fn):
            tid = tid_fn()
    except Exception:  # noqa: BLE001
        tid = None
    try:
        url_fn = getattr(langfuse, "get_trace_url", None)
        if callable(url_fn):
            url = url_fn()
    except Exception:  # noqa: BLE001
        url = None

    if tid:
        trace.trace_id = str(tid)
    if url:
        trace.public_url = str(url)
    elif tid:
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
        # Fallback path; project-scoped URLs are preferred via get_trace_url().
        trace.public_url = f"{host}/trace/{tid}"


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    as_type: str = "span",
) -> Iterator[SpanRecord]:
    """
    Open a nested span under the current root trace.

    When called outside an active root, creates a one-off local trace so
    standalone tool demos still produce a record.
    """
    parent = _CURRENT_SPAN.get()
    root = _ROOT_TRACE.get()
    created_root = False

    if root is None:
        root = TraceRecord(name=name)
        _ROOT_TRACE.set(root)
        created_root = True

    span = SpanRecord(
        name=name,
        parent_id=parent.span_id if parent else None,
        input=_jsonable(input),
        metadata=dict(metadata or {}),
    )

    if parent is None:
        root.root_spans.append(span)
    else:
        parent.children.append(span)

    token = _CURRENT_SPAN.set(span)
    langfuse = _get_langfuse()
    lf_cm = None
    lf_obs = None

    try:
        if langfuse is not None:
            try:
                lf_cm = _start_langfuse_span(
                    langfuse,
                    name,
                    input=input,
                    metadata=metadata,
                    as_type=as_type,
                )
                lf_obs = lf_cm.__enter__()
                if created_root:
                    _capture_langfuse_url(langfuse, root)
            except Exception as exc:  # noqa: BLE001
                print(f"[tracing] Langfuse span '{name}' failed: {exc}")
                lf_cm = None
                lf_obs = None

        yield span

        if lf_obs is not None:
            try:
                lf_obs.update(output=_jsonable(span.output))
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:
        span.error = str(exc)
        if lf_obs is not None:
            try:
                lf_obs.update(output={"error": str(exc)}, level="ERROR")
            except Exception:  # noqa: BLE001
                pass
        raise
    finally:
        span.end_time = _utc_now()
        if created_root and langfuse is not None and lf_obs is not None:
            _capture_langfuse_url(langfuse, root)
        if lf_cm is not None:
            try:
                lf_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        _CURRENT_SPAN.reset(token)
        if created_root:
            root.end_time = _utc_now()
            if langfuse is not None:
                try:
                    langfuse.flush()
                except Exception:  # noqa: BLE001
                    pass
            _ROOT_TRACE.set(None)


@contextmanager
def root_trace(
    name: str = "job_search_pipeline",
    *,
    metadata: dict[str, Any] | None = None,
) -> Iterator[TraceRecord]:
    """One outer trace for the whole pipeline. All tool spans nest under it."""
    if _ROOT_TRACE.get() is not None:
        # Already inside a root — yield the existing one.
        yield _ROOT_TRACE.get()  # type: ignore[misc]
        return

    trace = TraceRecord(name=name, metadata=dict(metadata or {}))
    token = _ROOT_TRACE.set(trace)
    langfuse = _get_langfuse()
    lf_cm = None
    lf_obs = None

    try:
        if langfuse is not None:
            try:
                lf_cm = _start_langfuse_span(
                    langfuse,
                    name,
                    metadata=metadata,
                    as_type="span",
                )
                lf_obs = lf_cm.__enter__()
                # Name the Langfuse *trace* (not just the root span) and mark
                # it public so the get_trace_url() link works without login
                # (required for assignment submission).
                try:
                    langfuse.update_current_trace(
                        name=name,
                        metadata=dict(metadata or {}),
                        public=True,
                    )
                except TypeError:
                    try:
                        langfuse.update_current_trace(
                            name=name,
                            metadata=dict(metadata or {}),
                        )
                        # Older SDKs: try span helper if present.
                        setter = getattr(lf_obs, "update_trace", None)
                        if callable(setter):
                            setter(public=True)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
                _capture_langfuse_url(langfuse, trace)
            except Exception as exc:  # noqa: BLE001
                print(f"[tracing] Langfuse root failed: {exc}")
                lf_cm = None
                lf_obs = None

        yield trace

        if lf_obs is not None:
            try:
                lf_obs.update(output={"status": "completed"})
            except Exception:  # noqa: BLE001
                pass
            # Capture URL again before exiting the active span context.
            if langfuse is not None:
                _capture_langfuse_url(langfuse, trace)
    finally:
        trace.end_time = _utc_now()
        if lf_cm is not None:
            try:
                lf_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        if langfuse is not None:
            try:
                langfuse.flush()
            except Exception:  # noqa: BLE001
                pass
        _ROOT_TRACE.reset(token)


def current_trace() -> TraceRecord | None:
    return _ROOT_TRACE.get()


def format_trace_summary(trace: TraceRecord | None = None) -> str:
    trace = trace or _ROOT_TRACE.get()
    if trace is None:
        return "(no active trace)"
    payload = trace.to_dict()
    return json.dumps(payload, indent=2, default=str)
