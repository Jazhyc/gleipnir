"""Record every file a process opens under the traced roots (build Pod only).

Installed temporarily through a ``.pth`` file in the serving interpreter, so it
also runs in vLLM's spawned workers. Each process appends absolute paths to
``$GLEIPNIR_OPEN_TRACE/<pid>.txt``. Native ``dlopen`` calls are not seen; the
image ships the native runtime trees whole.
"""

import os
import sys

ROOTS = ("/workspace/gleipnir/", "/tmp/", "/root/", "/opt/")
_directory = os.environ.get("GLEIPNIR_OPEN_TRACE")
_seen: set[str] = set()
_busy = False


def _hook(event: str, args: tuple) -> None:
    global _busy
    if event != "open" or _busy or not args or isinstance(args[0], int):
        return
    try:
        path = os.path.abspath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if path in _seen or not path.startswith(ROOTS):
        return
    _seen.add(path)
    _busy = True
    try:
        fd = os.open(
            os.path.join(_directory, f"{os.getpid()}.txt"),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        os.write(fd, (path + "\n").encode())
        os.close(fd)
    finally:
        _busy = False


if _directory:
    sys.addaudithook(_hook)
