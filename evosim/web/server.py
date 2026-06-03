"""Servidor web do EvoSim — apenas biblioteca padrão (http.server).

Expõe uma API JSON e serve o frontend estático. Sobe um único gerenciador de
evolução em memória. Inicie com:

    python -m evosim.cli web        # abre http://localhost:8000

Sem dependências externas: o frontend traz seu próprio visualizador 3D.
"""
from __future__ import annotations

import glob
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ..aptidao.funcoes import listar_fitness
from ..criaturas.presets import listar_presets
from ..persistencia.serializacao import carregar
from . import sim_api
from .manager import GerenciadorEvolucao

_DIR_STATIC = os.path.join(os.path.dirname(__file__), "static")
_GER = GerenciadorEvolucao()
_MIME = {".html": "text/html", ".js": "application/javascript",
         ".css": "text/css", ".json": "application/json"}


def _saves_disponiveis() -> list:
    achados = []
    for base in ("runs", "configs"):
        for ext in ("json", "yaml", "yml"):
            achados.extend(sorted(glob.glob(os.path.join(base, f"*.{ext}"))))
    return achados


class Handler(BaseHTTPRequestHandler):
    # silencia o log padrão (muito verboso).
    def log_message(self, *_a):  # noqa: D401
        pass

    # --- utilidades ---------------------------------------------------
    def _json(self, obj, status=200) -> None:
        corpo = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _ler_corpo(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def _estatico(self, caminho: str) -> None:
        if caminho in ("", "/"):
            caminho = "/index.html"
        rel = caminho.lstrip("/")
        # o frontend referencia os arquivos sob o prefixo /static/, mas eles
        # já vivem em _DIR_STATIC — remove o prefixo para não duplicar a pasta.
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        alvo = os.path.normpath(os.path.join(_DIR_STATIC, rel))
        if not alvo.startswith(_DIR_STATIC) or not os.path.isfile(alvo):
            self._json({"erro": "não encontrado"}, 404)
            return
        with open(alvo, "rb") as fh:
            dados = fh.read()
        ext = os.path.splitext(alvo)[1]
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    # --- GET ----------------------------------------------------------
    def do_GET(self) -> None:
        rota = urlparse(self.path).path
        if rota == "/api/opcoes":
            self._json({
                "presets": listar_presets(),
                "fitness": listar_fitness(),
                "algoritmos": ["es", "ga"],
                "controladores": ["cpg", "mlp"],
                "saves": _saves_disponiveis(),
                "cpus": os.cpu_count() or 1,
            })
        elif rota == "/api/status":
            self._json(_GER.status())
        elif rota == "/api/frames":
            self._json(_GER.frames_status())
        elif rota.startswith("/api/"):
            self._json({"erro": "rota desconhecida"}, 404)
        else:
            self._estatico(rota)

    # --- POST ---------------------------------------------------------
    def do_POST(self) -> None:
        rota = urlparse(self.path).path
        corpo = self._ler_corpo()
        if rota == "/api/iniciar":
            self._json(_GER.iniciar(corpo))
        elif rota == "/api/parar":
            self._json(_GER.parar())
        elif rota == "/api/salvar":
            caminho = corpo.get("caminho") or "runs/web_save.json"
            os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
            self._json(_GER.salvar(caminho))
        elif rota == "/api/playback":
            self._json(_GER.playback(corpo.get("ambiente", {}),
                                     float(corpo.get("segundos", 8.0))))
        elif rota == "/api/playback_save":
            try:
                save = carregar(corpo["save_path"])
                self._json(sim_api.frames_de_save(
                    save, corpo.get("ambiente"), float(corpo.get("segundos", 8.0))))
            except Exception as e:
                self._json({"erro": str(e), "frames": []}, 200)
        elif rota == "/api/corrida":
            try:
                saves = [carregar(p) for p in corpo.get("saves", [])]
                if len(saves) < 2:
                    raise ValueError("Escolha ao menos 2 saves.")
                self._json(sim_api.rodar_corrida_web(
                    saves, corpo.get("ambiente"), float(corpo.get("segundos", 10.0))))
            except Exception as e:
                self._json({"erro": str(e), "frames": []}, 200)
        elif rota == "/api/caca":
            try:
                sc = carregar(corpo["cacador"])
                sp = carregar(corpo["presa"])
                self._json(sim_api.rodar_caca_web(
                    sc, sp, corpo.get("ambiente"), float(corpo.get("segundos", 10.0))))
            except Exception as e:
                self._json({"erro": str(e), "frames": []}, 200)
        else:
            self._json({"erro": "rota desconhecida"}, 404)


def iniciar_servidor(host: str = "127.0.0.1", port: int = 8000,
                     abrir_navegador: bool = True) -> None:
    servidor = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"EvoSim web em {url}  (Ctrl+C para parar)")
    if abrir_navegador:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor.")
    finally:
        servidor.server_close()
