from __future__ import annotations

# Import the global Pydantic configuration first
import pydantic_global_config

import os
import time

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from modules import timer
from modules import initialize_util
from modules import initialize
from threading import Thread
from modules_forge.initialization import initialize_forge
from modules_forge import main_thread


startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize_forge()

initialize.imports()

initialize.check_versions()

initialize.initialize()


class ErrorResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    error: str = Field(description="Error type name")
    detail: str | None = Field(default=None, description="Error details")
    body: str | None = Field(default=None, description="Error body")
    message: str = Field(description="Error message")


def _handle_exception(request: Request, e: Exception):
    error_information = vars(e)
    content = ErrorResponse(
        error=type(e).__name__,
        detail=error_information.get("detail", ""),
        body=error_information.get("body", ""),
        message=str(e),
    )
    return JSONResponse(
        status_code=int(error_information.get("status_code", 500)), 
        content=jsonable_encoder(content.model_dump())
    )


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only_worker():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts
    from spaces import SessionMiddleware

    app = FastAPI()
    app.add_middleware(SessionMiddleware)
    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(
        server_name=initialize_util.gradio_server_name(),
        port=cmd_opts.port or 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )


def webui_worker():
    from modules.shared_cmd_options import cmd_opts

    launch_api = cmd_opts.api

    from modules import shared, ui_tempdir, script_callbacks, ui, progress, ui_extra_networks

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None
        # Ensure gradio_auth_creds is a list of (str, str) tuples or None
        if gradio_auth_creds is not None:
            gradio_auth_creds = [
                (str(cred[0]), str(cred[1]))
                for cred in gradio_auth_creds
                if isinstance(cred, (tuple, list)) and len(cred) == 2 and all(isinstance(x, str) for x in cred)
            ]
            if not gradio_auth_creds:
                gradio_auth_creds = None

        auto_launch_browser = False
        if os.getenv('SD_WEBUI_RESTARTING') != '1':
            if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not cmd_opts.webui_is_non_local

        from modules_forge.forge_canvas.canvas import canvas_js_root_path

        with startup_timer.subcategory("gradio launch"):
            app, local_url, share_url = shared.demo.launch(
                share=cmd_opts.share,
                server_name=initialize_util.gradio_server_name(),
                server_port=cmd_opts.port,
                ssl_keyfile=cmd_opts.tls_keyfile,
                ssl_certfile=cmd_opts.tls_certfile,
                ssl_verify=cmd_opts.disable_tls_verify,
                debug=cmd_opts.gradio_debug,
                auth=gradio_auth_creds,
                inbrowser=auto_launch_browser,
                prevent_thread_lock=True,
                allowed_paths=cmd_opts.gradio_allowed_path + [canvas_js_root_path],
                app_kwargs={
                    "docs_url": "/docs",
                    "redoc_url": "/redoc",
                    "exception_handlers": {Exception: _handle_exception},
                },
                root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
            )
            startup_timer.record("demo launch")

            # Safely filter out CORS middleware
            app.user_middleware = [x for x in app.user_middleware if not (getattr(x, 'cls', None) and getattr(x.cls, '__name__', '') == 'CORSMiddleware')]
            startup_timer.record("configure CORS middleware")

            initialize_util.setup_middleware(app)
            startup_timer.record("setup middleware")

            progress.setup_progress_api(app)
            startup_timer.record("setup progress API")
            ui.setup_ui_api(app)
            startup_timer.record("setup UI API")

        if launch_api:
            create_api(app)

        ui_extra_networks.add_pages_to_demo(app)

        startup_timer.record("add APIs")

        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)

        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")

        try:
            while True:
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            server_command = "stop"

        if server_command == "stop":
            print("Stopping server...")
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            shared.demo.close()
            break

        # disable auto launch webui in browser for subsequent UI Reload
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')

        print('Restarting UI...')
        shared.demo.close()
        time.sleep(0.5)
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize.initialize_rest(reload_script_modules=True)


def api_only():
    Thread(target=api_only_worker, daemon=True).start()


def webui():
    Thread(target=webui_worker, daemon=True).start()


if __name__ == "__main__":
    from modules.shared_cmd_options import cmd_opts

    if cmd_opts.nowebui:
        api_only()
    else:
        webui()

    main_thread.loop()
