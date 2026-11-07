"""Run the container entry point with the network cut off at socket level. Any attempt to connect fails loudly, so a
successful run proves that nothing is downloaded or called at inference (RULES 2.4). Arguments are those of run_container.py:
  python -I -X utf8 scripts/check_offline.py --data <split folder> --asr <whisper dir> --llm <llm dir> --adapter none --limit 3 --out outputs/offline.jsonl
Local, already-cached models only: a model id that is not in the local cache fails here, which is the point."""
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

attempts: list[str] = []


def _blocked(*args, **kwargs):
    attempts.append(repr(args[:2]))
    raise OSError("network access is blocked by scripts/check_offline.py")


socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

from lifemate.container import main  # noqa: E402

if __name__ == "__main__":
    code = main()
    print(f"network connection attempts blocked: {len(attempts)}")
    sys.exit(code if not attempts else 2)
