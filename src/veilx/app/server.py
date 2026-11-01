"""Local-only HTTP serving for the research app.

Gradio's stock page loads a script from a public CDN (iframe-resizer). VEIL-X instead mounts the
Gradio app in its own FastAPI application. Every response carries a Content-Security-Policy that
restricts all scripts, styles, fonts, images and connections to the local origin, so the browser
makes **no** third-party request. ``no-store`` stops the browser caching uploaded images, and the
API docs are disabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veilx.app.config import AppConfig

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def csp(host: str, port: int) -> str:
    origin = f"{host}:{port}"
    return "; ".join(
        [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Gradio's bundled frontend
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: blob:",
            "font-src 'self' data:",
            f"connect-src 'self' ws://{origin} http://{origin}",
            "frame-ancestors 'self'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
    )


def local_only_headers(host: str, port: int) -> dict[str, str]:
    if host not in LOOPBACK:
        raise ValueError("refusing to serve on a non-loopback address")
    return {
        "Content-Security-Policy": csp(host, port),
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }


def build_asgi(demo: Any, cfg: AppConfig, port: int, temp_root: Path) -> Any:
    import gradio as gr
    from fastapi import FastAPI, Request

    headers = local_only_headers(cfg.server.host, port)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    async def _local_only(request: Request, call_next: Any) -> Any:
        resp = await call_next(request)
        for k, v in headers.items():
            resp.headers[k] = v
        return resp

    app.middleware("http")(_local_only)

    return gr.mount_gradio_app(
        app,
        demo,
        path="",
        server_name=cfg.server.host,
        server_port=port,
        show_api=False,
        allowed_paths=[str(temp_root)],
        max_file_size=cfg.inputs.max_bytes,
        ssr_mode=False,
        enable_monitoring=False,
        pwa=False,
        mcp_server=False,
    )
