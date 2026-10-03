"""
Background jobs as the UI sees them: the scrape and train jobs the session
started (trigger + poller each), the search-index and taste re-encode chains
(status from the DB and the queue table, never the session), the nav
indicator, the fresh start that refuses to run next to a job, and the log
viewer that shows what the jobs wrote.

Two bookkeeping styles live here on purpose. Scrape and train are one task
each, so the session that started one remembers its id and start time (the
"session job" helpers, keyed by `kind` = "training" or "scrape", which is also
the prefix of the session keys). The chains are many short tasks with a new
id per slice, so nothing about them may live in a session.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings import embeddings, reset, search
from ratings.models import Image, LogEntry
from ratings.views.common import nav_counts

# django-q kills a task at this age, so the UI gives up on a job it cannot
# find at the same moment the worker would have; a crashed worker must not
# leave the nav spinner on forever.
_CLUSTER_TIMEOUT_SECONDS = settings.Q_CLUSTER["timeout"]


# ── Session jobs: scrape and train ───────────────────────────────────────────


def _fmt_elapsed(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def _elapsed_from_session(request, kind: str) -> int | None:
    """Seconds since the {kind}_started_at stamp in the session, or None without one."""
    started_at_str = request.session.get(f"{kind}_started_at")
    if not started_at_str:
        return None
    started_at = datetime.fromisoformat(started_at_str)
    return int((datetime.now(UTC) - started_at).total_seconds())


def _clear_job_session(request, kind: str) -> None:
    """
    Forget the session's job of this kind. Callers only reach this for the
    session's own job (they compare ids first), so there is no guard here.
    """
    request.session.pop(f"{kind}_task_id", None)
    request.session.pop(f"{kind}_started_at", None)


def _job_task_stale(request, kind: str, task_id: str) -> bool:
    """
    True when the session's job is no longer running in django-q.

    Used on ordinary page loads so a finished or lost worker job does not leave
    the nav spinner stuck until someone happens to poll the status view.

    fetch() only returns completed tasks (written to django_q_task after the
    worker finishes). While the task is queued or running it lives in
    django_q_ormq and fetch() returns None — that does NOT mean the task is
    lost. Elapsed time is the discriminator: None from fetch() is only stale
    once the elapsed time exceeds the cluster timeout, at which point the
    worker would have killed it anyway.
    """
    from django_q.tasks import fetch

    task = fetch(task_id)
    elapsed = _elapsed_from_session(request, kind)
    if task is None:
        return elapsed is None or elapsed > _CLUSTER_TIMEOUT_SECONDS
    if task.stopped is not None:
        return True
    return elapsed is not None and elapsed > _CLUSTER_TIMEOUT_SECONDS


def _session_job(request, kind: str) -> tuple[str | None, str | None]:
    """
    (task id, formatted elapsed) of the session's running job of this kind,
    or (None, None); a stale entry is cleared on the way.

    django-q doesn't expose task progress over HTTP, so elapsed time is tracked
    via session storage on the web process side.
    """
    task_id = request.session.get(f"{kind}_task_id")
    if not task_id:
        return None, None
    if _job_task_stale(request, kind, task_id):
        _clear_job_session(request, kind)
        return None, None
    return task_id, _fmt_elapsed(_elapsed_from_session(request, kind))


def training_ctx(request) -> dict:
    """Training-progress context for the templates and the nav indicator."""
    task_id, elapsed = _session_job(request, "training")
    return {"active_task_id": task_id, "training_elapsed": elapsed}


def scrape_ctx(request) -> dict:
    """Scrape-progress context, the twin of training_ctx, so a page reload during a scrape resumes the poller."""
    task_id, elapsed = _session_job(request, "scrape")
    return {"active_scrape_task_id": task_id, "scrape_elapsed": elapsed}


def _start_session_job(request, kind: str, task_func: str) -> tuple[str, str | None]:
    """
    Enqueue the job unless the session's own job of this kind is still
    running; return (task id to poll, formatted elapsed).

    Only a task that fetch() returns as not yet stopped counts as running
    here; a None result (queued, running, or lost) lets the user start again,
    so a job lost to a worker restart never blocks the button.
    """
    from django_q.tasks import async_task, fetch

    existing_id = request.session.get(f"{kind}_task_id")
    if existing_id:
        task = fetch(existing_id)
        if task is not None and task.stopped is None:
            return existing_id, _fmt_elapsed(_elapsed_from_session(request, kind))
        _clear_job_session(request, kind)

    task_id = async_task(task_func)
    request.session[f"{kind}_task_id"] = task_id
    request.session[f"{kind}_started_at"] = timezone.now().isoformat()
    return task_id, "0s"


@dataclass(frozen=True)
class _JobPoll:
    """
    What one poll of a session job found.

    state is "lost" (the session's own job is gone past the cluster timeout;
    the session is already cleared), "pending" (keep polling shown_task_id,
    showing elapsed) or "done" (task is the finished django-q task; the
    session is cleared when it was this job).
    """

    state: str
    shown_task_id: str
    elapsed: int | None = None
    task: object | None = None


def _poll_session_job(request, kind: str, task_id: str) -> _JobPoll:
    """
    The one decision tree behind scrape_status and train_status.

    fetch() returns None while the task is still queued or running in OrmQ,
    so a None result is only treated as a lost worker after the cluster
    timeout. A poll for a task that is not the session's own (another tab,
    a stale page) keeps showing the session's job while that one runs, and
    never clears the session for it.
    """
    from django_q.tasks import fetch

    session_task = request.session.get(f"{kind}_task_id")
    is_own_job = session_task == task_id
    task = fetch(task_id)

    if task is None:
        if is_own_job:
            elapsed = _elapsed_from_session(request, kind)
            if elapsed is None or elapsed > _CLUSTER_TIMEOUT_SECONDS:
                _clear_job_session(request, kind)
                return _JobPoll("lost", task_id)
            return _JobPoll("pending", task_id, elapsed)
        elapsed = _elapsed_from_session(request, kind) if session_task else None
        return _JobPoll("pending", session_task or task_id, elapsed)

    if task.stopped is None:
        elapsed = _elapsed_from_session(request, kind) if is_own_job else None
        return _JobPoll("pending", task_id, elapsed)

    if is_own_job:
        _clear_job_session(request, kind)
    return _JobPoll("done", task_id, task=task)


def task_result_dict(task) -> dict:
    """
    The dict a job returned, or {} when a hard crash left a traceback string
    or nothing at all.
    """
    if isinstance(task.result, dict):
        return task.result
    return {}


def task_outcome(task) -> tuple[bool, str | None]:
    """
    (succeeded, error text) of a finished django-q task.

    success=True from django-q only means the worker returned without
    raising; run_scrape and run_train catch their own exceptions and return
    {"ok": False, "error": ...}, so a genuine success needs both flags. The
    error text falls back to the raw result (a traceback string from a hard
    crash, capped) and then to a fixed sentence, so the fragment never shows
    an empty error.
    """
    result = task_result_dict(task)
    ok = bool(task.success and result.get("ok", True))
    if ok:
        return True, None
    if result.get("error"):
        return False, str(result["error"])
    if task.result:
        return False, str(task.result)[:500]
    return False, "Task exited without a result."


@login_required
@require_POST
def trigger_scrape(request):
    """
    Enqueue a scrape via django-q and return a polling fragment.

    Scraping used to run synchronously in the request, on the assumption it was
    "fast enough for a normal request timeout." That stopped being true once the
    4chan scraper switched from reading two index pages to fetching every live
    thread — one rate-limited request per thread (~1s each), so a multi-board
    scrape now runs for many minutes. A synchronous request would blow past
    gunicorn's --timeout (300s) and the worker would be SIGKILLed mid-scrape. So,
    like training, it now runs in a background worker and the UI polls
    scrape_status. The session stores the task ID for the poller to watch.
    """
    task_id, elapsed = _start_session_job(request, "scrape", "ratings.tasks.run_scrape")
    return render(
        request, "ratings/_scrape_pending.html", {"task_id": task_id, "elapsed": elapsed}
    )


@login_required
def scrape_status(request, task_id: str):
    """
    Polling endpoint for the active scrape job (mirror of train_status).

    Returns a "pending" fragment while the worker runs and a "result" fragment
    once it completes; the session entry is cleared on completion.
    """
    poll = _poll_session_job(request, "scrape", task_id)
    if poll.state == "lost":
        return render(
            request,
            "ratings/_scrape_result.html",
            {"ok": False, "error": "Scrape task not found (worker may have restarted)."},
        )
    if poll.state == "pending":
        return render(
            request,
            "ratings/_scrape_pending.html",
            {"task_id": poll.shown_task_id, "elapsed": _fmt_elapsed(poll.elapsed)},
        )
    ok, error = task_outcome(poll.task)
    if not ok:
        return render(request, "ratings/_scrape_result.html", {"ok": False, "error": error})
    result = task_result_dict(poll.task)
    return render(
        request,
        "ratings/_scrape_result.html",
        {"ok": True, "total": result.get("total", 0), "counts": result.get("counts", {})},
    )


_TRAIN_ETA_RE = re.compile(r"ETA (\d+)s")


def _training_eta(request) -> str | None:
    """Return a human-friendly ETA to training completion, or None.

    django-q runs training in a separate worker process that can't write the
    web session, so the only worker→web channel is the LogEntry stream (the
    same path ``elapsed`` rides). ``core.brain.encode`` logs progress lines
    like "train: 50/200 encoded (12.3 img/s, ETA 12s)"; the encode pass
    dominates training wall-clock, so its ETA is a good proxy for time left.

    Two filters keep the reading honest:
    - ``timestamp__gte`` the run's start stamp, because _trim_logs only drops
      entries >48h, so without it a fresh run would read the *previous* run's
      final "ETA 0s" before logging anything of its own.
    - ``train: `` prefix, so we ignore the later classify_images encode pass
      (logged under "classify_images: ") — otherwise the ETA would count down
      to ~0, training would not finish, then the ETA would jump back up.
    """
    started_str = request.session.get("training_started_at")
    if not started_str:
        return None
    started_at = datetime.fromisoformat(started_str)
    message = (
        LogEntry.objects.filter(
            source="train",
            timestamp__gte=started_at,
            message__startswith="train: ",
            message__contains="ETA ",
        )
        .order_by("-pk")
        .values_list("message", flat=True)
        .first()
    )
    if not message:
        return None
    match = _TRAIN_ETA_RE.search(message)
    if not match:
        return None
    return _fmt_elapsed(int(match.group(1)))


def _render_train_pending(request, task_id: str, elapsed: str | None):
    """
    The one place that renders _train_pending.html, so every path (the
    trigger, the steady-state poll, the OrmQ-pending poll) shows the ETA.
    """
    ctx = {"task_id": task_id, "elapsed": elapsed, "eta": _training_eta(request)}
    return render(request, "ratings/_train_pending.html", ctx)


@login_required
@require_POST
def trigger_train(request):
    """
    Enqueue a training job via django-q and return a polling fragment.

    Training blocks for several minutes (DINOv3 encoding + LogReg fit), so it
    runs in a background worker. The session stores the task ID so the polling
    template knows which job to watch via train_status.
    """
    task_id, elapsed = _start_session_job(request, "training", "ratings.tasks.run_train")
    return _render_train_pending(request, task_id, elapsed)


@login_required
def train_status(request, task_id: str):
    """
    Polling endpoint for the active training job.

    Returns a "pending" fragment while the worker is running, and a "result"
    fragment once it completes. The session entry is cleared on completion so
    a subsequent visit to stats doesn't show a stale training indicator.
    """
    poll = _poll_session_job(request, "training", task_id)
    if poll.state == "lost":
        return render(
            request,
            "ratings/_train_result.html",
            {"ok": False, "error": "Training task not found (worker may have restarted)."},
        )
    if poll.state == "pending":
        return _render_train_pending(request, poll.shown_task_id, _fmt_elapsed(poll.elapsed))

    result = poll.task.result or {}
    return render(
        request,
        "ratings/_train_result.html",
        {
            "ok": result.get("ok", False),
            "error": result.get("error", "Unknown error."),
            # Disjoint counts matching the trainer's split: positives (score>=3)
            # vs below-cutoff negatives (score<=2). Together they're the full
            # training set, so the "+" in the message doesn't double-count.
            "positive_n": Image.objects.filter(score__gte=3).count(),
            "negative_n": Image.objects.filter(score__lte=2).count(),
        },
    )


# ── Chains: search index and taste re-encode ─────────────────────────────────


def index_status_ctx() -> dict:
    """
    Everything the index block on the config page needs, read from the
    database (counts) and the queue table (queued flag), never from the
    session (contract V15). The report about the last slice only matters when
    nothing is queued and rows are still waiting: while the chain runs the
    pending fragment is the status, and when it is done there is nothing to
    report.
    """
    indexed, total = search.index_counts()
    remaining = total - indexed
    queued = search.index_job_queued()
    report = None
    if not queued and remaining > 0:
        report = search.last_index_report()
    return {
        "indexed": indexed,
        "total": total,
        "remaining": remaining,
        "index_queued": queued,
        "index_report": report,
    }


def _render_index_status(request):
    ctx = index_status_ctx()
    if ctx["index_queued"]:
        return render(request, "ratings/_index_pending.html", ctx)
    return render(request, "ratings/_index_result.html", ctx)


@login_required
@require_POST
def trigger_search_index(request):
    """
    Start the search index chain and return its status fragment.

    Unlike trigger_scrape / trigger_train there is no session bookkeeping: the
    chain is many short tasks, and enqueue_index_job_if_needed already refuses
    a second chain (V13), so a double tap just re-renders the pending state.
    """
    search.enqueue_index_job_if_needed()
    return _render_index_status(request)


@login_required
def search_index_status(request):
    """Polling target of _index_pending.html; the fragment it returns decides whether polling goes on."""
    return _render_index_status(request)


def reencode_status_ctx() -> dict:
    """
    Everything the taste-vector block on the config page needs (taste contract
    V21), read from the database and the queue table like index_status_ctx,
    never from the session: the chain is many short tasks with new ids.
    """
    current, total = embeddings.taste_vector_counts()
    remaining = total - current
    queued = embeddings.reencode_job_queued()
    report = None
    if not queued and remaining > 0:
        report = embeddings.last_reencode_report()
    return {
        "reencoded": current,
        "reencode_total": total,
        "reencode_remaining": remaining,
        "reencode_queued": queued,
        "reencode_report": report,
    }


def _render_reencode_status(request):
    ctx = reencode_status_ctx()
    if ctx["reencode_queued"]:
        return render(request, "ratings/_reencode_pending.html", ctx)
    return render(request, "ratings/_reencode_result.html", ctx)


@login_required
@require_POST
def trigger_taste_reencode(request):
    """
    Start the taste re-encode chain and return its status fragment (V20, V21).

    Like trigger_search_index there is no session bookkeeping, and
    enqueue_reencode_job_if_needed refuses a second chain, so a double tap
    just re-renders the pending state.
    """
    embeddings.enqueue_reencode_job_if_needed()
    return _render_reencode_status(request)


@login_required
def taste_reencode_status(request):
    """Polling target of _reencode_pending.html; the fragment it returns decides whether polling goes on."""
    return _render_reencode_status(request)


# ── Nav indicator and fresh start ────────────────────────────────────────────


@login_required
def job_indicator(request):
    """
    Polling target for the nav's job indicator (UI contract V8).

    The context comes from the active_jobs context processor, so this view
    only picks the partial; it is polled every few seconds while a job runs
    and the partial stops polling itself once it renders the idle state.
    """
    return render(request, "ratings/_nav_job.html")


def _job_is_running(request) -> bool:
    """
    True while a scrape, train or chain job is queued or running anywhere.

    The session knows about jobs this browser started; OrmQ (django-q's ORM
    broker table) still holds every task that is queued or running, including
    scheduled scrapes and jobs started from another session, until the worker
    acknowledges it after completion.
    """
    from django_q.models import OrmQ

    if training_ctx(request)["active_task_id"]:
        return True
    if scrape_ctx(request)["active_scrape_task_id"]:
        return True
    return OrmQ.objects.exists()


@login_required
@require_POST
def fresh_start_view(request):
    """
    Start over: wipe the library, keep the configuration (UI contract V12).

    Three guards, all enforced here and not only in the UI: login and POST via
    the decorators (OWASP A01), no running scrape or train job (its worker
    would write rows and files into the wiped library), and the exact
    confirmation word in the body (OWASP A04: a confirmation that lives only in
    the sheet is one crafted request away from nothing).
    """
    if _job_is_running(request):
        messages.error(
            request,
            "A scrape, train, index or re-encode job is running. Wait for it to finish before starting over.",
        )
        return redirect("config")
    if request.POST.get("confirm", "").strip() != reset.CONFIRM_WORD:
        messages.error(
            request, f"Type {reset.CONFIRM_WORD} to confirm the fresh start. Nothing was deleted."
        )
        return redirect("config")
    result = reset.fresh_start(
        Path(settings.DATA_DIR),
        [Path(settings.WEIGHTS_PATH), Path(settings.NSFW_WEIGHTS_PATH)],
    )
    messages.success(
        request,
        f"Fresh start done: {result['images']} images and {result['weights']} classifier "
        "file(s) removed. Sources, channels and settings kept.",
    )
    return redirect("config")


# ── Log viewer ───────────────────────────────────────────────────────────────

_LOG_SOURCES = {"scrape", "train"}


def _log_source_filter(request) -> str | None:
    """Return the ?source= filter if it matches a known source, else None for 'all'."""
    src = request.GET.get("source", "").strip().lower()
    return src if src in _LOG_SOURCES else None


@login_required
def logs_page(request):
    """
    Show the 500 most recent log entries, optionally filtered by source.

    Source filter is a query param (?source=train|scrape) so it survives an
    HTMX swap of the entries fragment — the polling endpoint reads the same
    value and only returns entries matching the active source.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    source = _log_source_filter(request)
    qs = LogEntry.objects.all()
    if source:
        qs = qs.filter(source=source)
    entries = list(qs.order_by("-pk")[:500])
    next_since = entries[0].pk if entries else 0
    return render(
        request,
        "ratings/logs.html",
        {
            **nav_counts(show_nsfw),
            "show_nsfw": show_nsfw,
            "entries": entries,
            "next_since": next_since,
            "active_source": source or "all",
            "mode": "logs",
        },
    )


@login_required
def log_entries(request):
    """Polling endpoint for log updates; returns only entries newer than since_id."""
    try:
        since_id = int(request.GET.get("since", 0))
    except (ValueError, TypeError):
        since_id = 0
    source = _log_source_filter(request)
    qs = LogEntry.objects.filter(pk__gt=since_id)
    if source:
        qs = qs.filter(source=source)
    entries = list(qs.order_by("-pk")[:100])
    next_since = entries[-1].pk if entries else since_id
    return render(
        request,
        "ratings/_log_entries.html",
        {
            "entries": entries,
            "next_since": next_since,
            "active_source": source or "all",
        },
    )


@login_required
@require_POST
def log_clear(request):
    """Truncate all log entries — useful before a scrape to keep the log view clean."""
    LogEntry.objects.all().delete()
    messages.success(request, "Log cleared")
    return redirect("logs")
