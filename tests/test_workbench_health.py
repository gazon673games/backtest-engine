"""Readiness remains available when saved history cannot be loaded."""

import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

from backtest_engine.workbench import server


def test_health_does_not_access_history_or_catalogs(tmp_path, monkeypatch):
    def unavailable(*_args):
        raise AssertionError("Readiness must not load saved runs or catalogs")

    monkeypatch.setattr(server, "load_workbench_history", unavailable)
    monkeypatch.setattr(server, "get_agent_payload", unavailable)
    for route in server._GET_HANDLERS:
        monkeypatch.setitem(server._GET_HANDLERS, route, unavailable)
    (tmp_path / "history.json").write_text("invalid history", encoding="utf-8")
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class(tmp_path / "runs"))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", http.server_port, timeout=2)
    try:
        connection.request("GET", "/api/health")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        assert json.loads(response.read()) == {
            "service": "ohlcv-replay-workbench",
            "status": "ready",
        }
    finally:
        connection.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
