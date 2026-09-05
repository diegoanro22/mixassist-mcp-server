"""El arranque por línea de comandos.

El modo red salió roto en v0.1.0: `main()` escribía el host y el puerto en
`server.settings`, que es API del SDK 1.x y en el 2.x ya no existe, así que el
proceso moría con `ValueError: "Settings" object has no field "host"` antes de
llegar a escuchar. Ninguno de los tests lo notó porque todos ejercitan las
tools directamente y nunca pasan por aquí.
"""

import pytest

from mixassist_mcp import server as module


@pytest.fixture
def captured_run(monkeypatch):
    """Reemplaza server.run() y devuelve los argumentos con que se llamó."""
    calls = []
    monkeypatch.setattr(module.server, "run", lambda **kwargs: calls.append(kwargs))
    return calls


def run_cli(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["mixassist-mcp", *args])
    module.main()


def test_por_defecto_arranca_en_stdio(monkeypatch, captured_run):
    run_cli(monkeypatch)

    assert captured_run == [{"transport": "stdio"}]


def test_stdio_no_recibe_host_ni_puerto(monkeypatch, captured_run):
    """El transporte stdio no escucha en ningún lado; pasárselos sería un error."""
    run_cli(monkeypatch, "--transport", "stdio", "--host", "0.0.0.0", "--port", "9000")

    assert captured_run == [{"transport": "stdio"}]


def test_streamable_http_reenvia_host_y_puerto(monkeypatch, captured_run):
    run_cli(
        monkeypatch,
        "--transport", "streamable-http",
        "--host", "0.0.0.0",
        "--port", "8077",
    )

    assert captured_run == [
        {"transport": "streamable-http", "host": "0.0.0.0", "port": 8077}
    ]


def test_sse_tambien_los_reenvia(monkeypatch, captured_run):
    run_cli(monkeypatch, "--transport", "sse", "--host", "127.0.0.1", "--port", "8000")

    assert captured_run == [{"transport": "sse", "host": "127.0.0.1", "port": 8000}]


def test_los_valores_por_defecto_de_red_son_locales(monkeypatch, captured_run):
    """Sin --host explícito no se escucha hacia afuera."""
    run_cli(monkeypatch, "--transport", "streamable-http")

    assert captured_run == [
        {"transport": "streamable-http", "host": "127.0.0.1", "port": 8000}
    ]


def test_un_transporte_inventado_se_rechaza(monkeypatch, captured_run):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--transport", "carrier-pigeon")

    assert captured_run == []


def test_run_no_toca_settings(monkeypatch, captured_run):
    """La regresión concreta: Settings del SDK 2.x no tiene host ni port."""
    run_cli(monkeypatch, "--transport", "streamable-http", "--port", "8077")

    assert not hasattr(module.server.settings, "host")
    assert not hasattr(module.server.settings, "port")
