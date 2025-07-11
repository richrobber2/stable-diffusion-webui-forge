from __future__ import annotations
import base64
import io
import time
import threading
from collections import deque

import gradio as gr
from pydantic import BaseModel, Field, ConfigDict
from typing import List

from modules.shared import opts

import modules.shared as shared
from collections import OrderedDict
import string
import random
from fastapi import WebSocket

current_task = None
pending_tasks = deque()
finished_tasks = []
recorded_results = []
recorded_results_limit = 2
# Add a dictionary to hold events for each task
_task_events = {}


def start_task(id_task):
    global current_task

    current_task = id_task
    try:
        pending_tasks.remove(id_task)
    except ValueError:
        pass
    # Ensure an event exists for this task
    if id_task not in _task_events:
        _task_events[id_task] = threading.Event()


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)
    # Signal the event for this task
    if id_task in _task_events:
        _task_events[id_task].set()

def create_task_id(task_type):
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase +
    string.digits, k=N))
    return f"task({task_type}-{res})"

def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def add_task_to_queue(id_job):
    pending_tasks.append(id_job)

class PendingTasksResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    size: int = Field(title="Pending task size")
    tasks: List[str] = Field(title="Pending task ids")

class ProgressRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image") 
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")

class ProgressResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: float | None = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float | None = Field(default=None, title="ETA in secs")
    live_preview: str | None = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int | None = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: str | None = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)

    # --- WebSocket endpoint for real-time progress and live preview updates ---
    from fastapi import WebSocket, WebSocketDisconnect
    import asyncio

    app.progress_ws_clients = set()

    @app.websocket("/ws/progress/{id_task}")
    async def progress_ws(websocket: WebSocket, id_task: str):
        await websocket.accept()
        app.progress_ws_clients.add(websocket)
        try:
            last_progress = None
            last_id_live_preview = None
            while True:
                # Get current progress for this task
                active = id_task == current_task
                queued = id_task in pending_tasks
                completed = id_task in finished_tasks
                progress = 0
                job_count, job_no = shared.state.job_count, shared.state.job_no
                sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step
                if job_count > 0:
                    progress += job_no / job_count
                if sampling_steps > 0 and job_count > 0:
                    progress += 1 / job_count * sampling_step / sampling_steps
                progress = min(progress, 1)

                # Live preview logic
                live_preview = None
                id_live_preview = shared.state.id_live_preview
                if opts.live_previews_enable:
                    shared.state.set_current_image()
                    image = shared.state.current_image
                    if image is not None and last_id_live_preview != id_live_preview:
                        buffered = io.BytesIO()
                        if opts.live_previews_image_format == "png":
                            if max(*image.size) <= 256:
                                save_kwargs = {"optimize": True}
                            else:
                                save_kwargs = {"optimize": False, "compress_level": 1}
                        else:
                            save_kwargs = {}
                        image.save(buffered, format=opts.live_previews_image_format, **save_kwargs)
                        base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                        live_preview = f"data:image/{opts.live_previews_image_format};base64,{base64_image}"
                        last_id_live_preview = id_live_preview

                # Only send if changed
                if last_progress != progress or completed or live_preview is not None:
                    await websocket.send_json({
                        "active": active,
                        "queued": queued,
                        "completed": completed,
                        "progress": progress,
                        "live_preview": live_preview,
                        "id_live_preview": id_live_preview,
                    })
                    last_progress = progress
                if completed:
                    break
                await asyncio.sleep(0.2)  # Tune for responsiveness
        except WebSocketDisconnect:
            pass
        finally:
            app.progress_ws_clients.discard(websocket)


def get_pending_tasks():
    pending_tasks_ids = list(pending_tasks)
    pending_len = len(pending_tasks_ids)
    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


def progressapi(req: ProgressRequest):
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks

    if not active:
        textinfo = "Waiting..."
        if queued:
            queue_index = list(pending_tasks).index(req.id_task)
            textinfo = f"In queue: {queue_index + 1}/{len(pending_tasks)}"
        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    progress = 0

    job_count, job_no = shared.state.job_count, shared.state.job_no
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)

    time_start = shared.state.time_start or time.time()  # Use current time if time_start is None
    elapsed_since_start = time.time() - time_start
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    live_preview = None
    id_live_preview = req.id_live_preview

    if opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image()
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            if image is not None:
                buffered = io.BytesIO()

                if opts.live_previews_image_format == "png":
                    # using optimize for large images takes an enormous amount of time
                    if max(*image.size) <= 256:
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}

                else:
                    save_kwargs = {}

                image.save(buffered, format=opts.live_previews_image_format, **save_kwargs)
                base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                live_preview = f"data:image/{opts.live_previews_image_format};base64,{base64_image}"
                id_live_preview = shared.state.id_live_preview

    return ProgressResponse(active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)


def restore_progress(id_task):
    # Wait for the event to be set, or for the task to be out of current/pending
    event = _task_events.get(id_task)
    if event is not None:
        event.wait()
    else:
        # fallback: previous logic if no event (should not happen)
        while id_task == current_task or id_task in pending_tasks:
            time.sleep(0.1)

    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
