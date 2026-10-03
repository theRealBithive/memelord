"""
The "Classify now" job and the numbers behind the NSFW head (NSFW contract
N5, N8).

One task, not a chain: classify_images reads stored vectors and never encodes,
so even the whole unrated backlog is a matter of minutes, well inside the
cluster timeout. Queue state comes from OrmQ like the index and re-encode
chains, never from the session, so a second browser and the second gunicorn
worker see the same thing.
"""

from __future__ import annotations

from ratings.models import Image

CLASSIFY_TASK = "ratings.tasks.run_classify"


def classify_job_queued() -> bool:
    """
    True while a classify run is queued or running.

    OrmQ is django-q's queue table; a task's row stays there until the worker
    acknowledges it after completion, so "queued or running" is one question.
    """
    from django_q.models import OrmQ

    return any(queued.func() == CLASSIFY_TASK for queued in OrmQ.objects.all())


def enqueue_classify_job() -> bool:
    """
    Queue one classify run unless one is already queued (N5: never twice).

    No "nothing to do" check: whether the heads exist is the run's business,
    and its report says so on the config page.
    """
    if classify_job_queued():
        return False
    from django_q.tasks import async_task

    async_task(CLASSIFY_TASK)
    return True


def last_classify_report() -> dict | None:
    """
    The outcome of the last finished run as {"ok", "text"}, or None before
    the first run; shown on the config page while no run is queued (N5).
    """
    from django_q.models import Task

    task = Task.objects.filter(func=CLASSIFY_TASK).order_by("-stopped").first()
    if task is None:
        return None
    result = task.result if isinstance(task.result, dict) else {}
    if not task.success or result.get("ok") is False:
        error = result.get("error") or str(task.result or "Task exited without a result.")
        return {"ok": False, "text": f"Last classify run failed: {error}"}
    processed = result.get("processed", 0)
    tagged = result.get("nsfw_tagged", 0)
    untagged = result.get("nsfw_untagged", 0)
    text = f"Classified {processed} unrated images: {tagged} flagged, {untagged} unflagged by the NSFW head."
    return {"ok": True, "text": text}


def nsfw_head_counts() -> dict:
    """
    How many decided images train the head on each side, and how many flags
    the model set that nobody has decided yet (N8). Index-only via
    image_nsfw_judged_idx.
    """
    live = Image.objects.filter(is_purged=False)
    return {
        "nsfw_flagged_judged": live.filter(nsfw_judged=True, is_nsfw=True).count(),
        "nsfw_safe_judged": live.filter(nsfw_judged=True, is_nsfw=False).count(),
        "nsfw_undecided_flags": live.filter(nsfw_judged=False, is_nsfw=True).count(),
    }
