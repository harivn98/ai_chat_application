"""Terminal output of the evaluation: plain lines, and status lines that update in place while a step runs."""
import sys
import threading
import time

_TTY = sys.stdout.isatty()


def say(text: str = "") -> None:
    print(text, flush=True)


def short(text: str, n: int = 70) -> str:
    """The text on one line, cut to n characters."""
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def duration(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"


class Status:
    """A status line that keeps updating in place (with elapsed time) until done() is called."""

    def __init__(self, label: str):
        self.label, self.detail, self.t0 = label, "", time.perf_counter()
        self._width, self._stopped, self._lock = 0, False, threading.Lock()
        if _TTY:
            threading.Thread(target=self._tick, daemon=True).start()
        else:
            say(label)

    def _tick(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                if self._stopped:
                    return
                self._write(f"{self.label} {self.detail} [{duration(self.elapsed)}]")

    def _write(self, text: str):
        sys.stdout.write("\r" + text.ljust(self._width))
        sys.stdout.flush()
        self._width = len(text)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def set(self, detail: str):
        self.detail = detail

    def done(self, text: str):
        with self._lock:
            self._stopped = True
            if _TTY:
                self._write(text)
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                say(text)
