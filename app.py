import sys
import uuid
import signal
import asyncio
from aioconsole import ainput
from config.settings import load_config
from config.logger import setup_logging
from core.utils.util import get_local_ip
from core.http_server import SimpleHttpServer
from core.websocket_server import WebSocketServer
from core.utils.util import check_ffmpeg_installed

TAG = __name__
logger = setup_logging()


async def wait_for_exit() -> None:

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    if sys.platform != "win32":  # Unix / macOS
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
        await stop_event.wait()
    else:
        try:
            await asyncio.Future()
        except KeyboardInterrupt:  # Ctrl‑C
            pass

async def monitor_stdin():

    while True:
        await ainput()

async def main():
    check_ffmpeg_installed()
    config = load_config()

    auth_key = str(uuid.uuid4().hex)
    config["server"]["auth_key"] = auth_key

    stdin_task = asyncio.create_task(monitor_stdin())

    # Start WebSocket server
    ws_server = WebSocketServer(config)
    ws_task = asyncio.create_task(ws_server.start())
    # Start Simple http server for OTA
    ota_server = SimpleHttpServer(config)
    ota_task = asyncio.create_task(ota_server.start())

    port = int(config["server"].get("http_port", 8003))

    logger.bind(tag=TAG).info(
        "OTA:\thttp://{}:{}/ai/ota/",
        get_local_ip(),
        port,
    )

    # Get WebSocket configuration，using default for now.
    websocket_port = 8000
    server_config = config.get("server", {})
    if isinstance(server_config, dict):
        websocket_port = int(server_config.get("port", 8000))

    logger.bind(tag=TAG).info(
        "Websocket address:\tws://{}:{}/ai/v1/",
        get_local_ip(),
        websocket_port,
    )

    logger.bind(tag=TAG).info(
        "=============================================================\n"
    )

    try:
        await wait_for_exit()
    except asyncio.CancelledError:
        print("The task was canceled, cleaning up resources...")
    finally:
        # Cancel all tasks
        stdin_task.cancel()
        ws_task.cancel()
        if ota_task:
            ota_task.cancel()

        # Wait for all tasks to finish
        await asyncio.wait(
            [stdin_task, ws_task, ota_task] if ota_task else [stdin_task, ws_task],
            timeout=3.0,
            return_when=asyncio.ALL_COMPLETED,
        )
        print("The server has been shut down and the program has exited.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Manually interrupt the program.")
