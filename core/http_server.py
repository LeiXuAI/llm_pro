import asyncio
from aiohttp import web
from config.logger import setup_logging
from core.api.ota_handler import OTAHandler

TAG = __name__


class SimpleHttpServer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.ota_handler = OTAHandler(config)

    def _get_websocket_url(self, local_ip: str, port: int) -> str:

        server_config = self.config["server"]
        websocket_config = server_config.get("websocket")
        # Use default WebSocket URL for now.
        return f"ws://{local_ip}:{port}/ai/v1/"

    async def start(self):
        server_config = self.config["server"]
        host = server_config.get("ip", "0.0.0.0")
        port = int(server_config.get("http_port", 8003))

        if port:
            app = web.Application()

            app.add_routes(
                [
                    web.get("/ai/ota/", self.ota_handler.handle_get),
                    web.post("/ai/ota/", self.ota_handler.handle_post),
                    web.options("/ai/ota/", self.ota_handler.handle_post),
                ]
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()

            while True:
                await asyncio.sleep(3600)
