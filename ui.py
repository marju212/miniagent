"""
ui - everything miniagent puts on the screen.

Kept apart from agent.py so the loop can be read without the terminal
handling, and so this can be tested on its own.  It imports nothing from
agent.py or policy.py; that is what keeps the split acyclic.

Ordinary lines are printed.  The escape sequences that move the cursor go
through emit() instead, which holds a re-entrant lock.  The reason is
DECSC/DECRC (\0337 / \0338): a terminal has *one* cursor save slot, so two
writers interleaving a save/restore pair corrupt each other.  The status bar
draws from the prompt loop and gives its row back from a signal handler, which
can fire in the middle of that draw - the lock is what keeps the two apart,
and it is re-entrant because a handler runs on the thread it interrupted.
"""

import atexit
import codecs
import difflib
import json
import os
import re
import select
import shutil
import signal
import sys
import threading

try:                       # raw keyboard input, for reading a bare escape
    import termios
    import tty
except ImportError:        # not POSIX
    termios = tty = None


# ---------------------------------------------------------------- output
_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

# One writer at a time; see the module docstring.
OUT = threading.RLock()


def emit(s: str) -> None:
    """Write one complete, self-contained piece of terminal output.

    Never raises: this also runs from the SIGHUP handler, which is delivered
    precisely because the terminal has gone away, and a failed write there
    would replace the intended exit status with a traceback.
    """
    with OUT:
        try:
            sys.stdout.write(s)
            sys.stdout.flush()
        except (OSError, ValueError):       # terminal gone, or stream closed
            pass


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def dim(s): return _c("2", s)
def bold(s): return _c("1", s)
def red(s): return _c("31", s)
def green(s): return _c("32", s)
def yellow(s): return _c("33", s)
def cyan(s): return _c("36", s)


def warn(s: str) -> None:
    print(yellow(f"  ! {s}"), file=sys.stderr)


def _short(s: str, n: int = 100) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "..."


def _clip(text: str, limit: int) -> str:
    """Keep both ends of an oversized tool result - the tail usually has the
    error message, the head usually has the shape of the output."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    cut = len(text) - limit
    return f"{text[:head]}\n...[{cut} chars cut from the middle]...\n{text[-tail:]}"


# ---------------------------------------------------------------- status bar
class StatusBar:
    """A line pinned to the bottom row of the terminal.

    The last row is taken out of the scrolling region, so everything the agent
    prints scrolls above it and the bar stays where it is - visible while tools
    are running, not only at the prompt.
    """

    def __init__(self):
        self.on = False
        self.stale = False
        self.text = ""

    def install(self) -> bool:
        if not _TTY or os.environ.get("AGENT_STATUS") == "off":
            return False
        if os.environ.get("TERM", "dumb") == "dumb":
            return False
        rows = shutil.get_terminal_size().lines
        if rows < 3:
            return False
        # Make room for the bar. At the bottom of the screen this scrolls;
        # anywhere else stepping back up undoes it.
        # DECSTBM homes the cursor, so save and restore it around the change.
        emit(f"\n\033[1A\0337\033[1;{rows - 1}r\0338")
        self.on = True
        atexit.register(self.remove)
        try:
            signal.signal(signal.SIGWINCH, self._resized)
        except (AttributeError, ValueError, OSError):
            pass                    # no SIGWINCH here, or not the main thread
        self.draw()
        return True

    def _resized(self, *_a) -> None:
        # Only a flag: writing to stdout from a signal handler can re-enter a
        # half-finished write and raise, and DECSC has one save slot per
        # terminal, so a save here would clobber an outer draw's. The region is
        # re-cut at the next draw instead.
        self.stale = True

    def draw(self, text: str = None) -> None:
        if text is not None:
            self.text = text
        if not self.on:
            return
        size = shutil.get_terminal_size()
        recut = ""
        if self.stale:                       # the terminal was resized
            self.stale = False
            if size.lines < 3:
                self.remove()
                return
            recut = f"\0337\033[1;{size.lines - 1}r\0338"
        # not _short(): it collapses runs of spaces, and the gaps between the
        # fields are what makes the bar readable
        room = max(1, size.columns - 2)
        line = self.text
        if len(line) > room:
            line = (line[:max(0, room - 3)] + "...")[:room]
        # One write: a second writer must never land between the save and the
        # restore.
        emit(f"{recut}\0337\033[{size.lines};1H\033[2K"
             f"\033[90m {line}\033[0m\0338")

    def remove(self) -> None:
        if not self.on:
            return
        self.on = False
        rows = shutil.get_terminal_size().lines
        emit(f"\0337\033[r\033[{rows};1H\033[2K\0338")


BAR = StatusBar()


# ---------------------------------------------------------------- the prompt
def prompt_text(text: str) -> str:
    """A prompt readline can measure.

    It counts the prompt's width to know where to wrap and redraw an edited
    line, so the colour codes have to be marked as taking up no space.
    """
    if not _TTY:
        return text
    return f"\001\033[1m\002{text}\001\033[0m\002"


class Interrupted(Exception):
    """The user pressed escape at a permission prompt."""


def _pending(fd, timeout: float = 0.05) -> bool:
    return bool(select.select([fd], [], [], timeout)[0])


def read_answer(prompt: str):
    """A short answer from the terminal, or None if escape was pressed.

    `input()` cannot see escape: it is line buffered, so a bare keypress never
    arrives. Reading the descriptor directly is the only way to notice it -
    and it has to be `os.read`, because a buffered reader would swallow the
    rest of an arrow key before select() could tell it apart from escape.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if termios is None or not sys.stdin.isatty():
        try:
            return input()
        except EOFError:
            print()
            return ""

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    decode = codecs.getincrementaldecoder("utf-8")("replace").decode
    typed, pushed = [], []

    def getch() -> str:
        """One character, or "" at end of input. Never a partial one."""
        if pushed:
            return pushed.pop()
        while True:
            raw = os.read(fd, 1)
            if not raw:
                return ""
            ch = decode(raw)
            if ch:
                return ch

    try:
        # TCSAFLUSH: anything typed before the prompt appeared is dropped
        # rather than allowed to answer a permission question by accident.
        tty.setcbreak(fd)
        while True:
            ch = getch()
            if not ch:                        # end of input
                print()
                return ""
            if ch == "\x1b":
                if not _pending(fd):          # nothing follows: a real escape
                    print()
                    return None
                # An arrow key and friends. Consume exactly the sequence -
                # draining everything available would eat the line behind it.
                nxt = getch()
                if nxt == "[":                # CSI: parameters, then a final byte
                    while True:
                        end = getch()
                        if not end or "@" <= end <= "~":
                            break
                elif nxt == "O":              # SS3: one byte follows
                    getch()
                elif nxt:
                    pushed.append(nxt)        # not a sequence we know
                continue
            if ch in ("\r", "\n"):
                print()
                return "".join(typed)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x04":
                print()
                return ""
            if ch in ("\x7f", "\b"):
                if typed:
                    typed.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch < " ":
                continue
            typed.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# ---------------------------------------------------------------- diffs
def _fit(line: str, room: int) -> str:
    """Cut a line to the terminal width.  Diffs are never wrapped: a wrapped
    diff loses the one thing that makes it readable, which is that the first
    column says what happened to the line."""
    return line if len(line) <= room else line[:max(0, room - 1)] + "…"


def diff_lines(before: str, after: str, max_lines: int = 24) -> list:
    """A coloured unified diff, and an honest count of what was left out."""
    body = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                     lineterm="", n=3))
    if not body:
        return [dim("  (no change)")]
    # Drop the two file headers by position. Matching them by prefix would
    # also eat a real removed line reading `--- x` or an added `+++ x`, which
    # is how a deleted `-- comment` came to be shown as no change at all.
    body = body[2:]
    # Count over the whole diff before truncating, so the summary is true.
    plus = sum(1 for l in body if l.startswith("+"))
    minus = sum(1 for l in body if l.startswith("-"))
    room = max(20, shutil.get_terminal_size().columns - 2)

    out, shown = [], body[:max_lines] if max_lines else []
    for l in shown:
        l = _fit(l, room)
        if l.startswith("+"):
            out.append("  " + green(l))
        elif l.startswith("-"):
            out.append("  " + red(l))
        elif l.startswith("@@"):
            out.append("  " + cyan(l))
        else:
            out.append("  " + dim(l))
    rest = len(body) - len(shown)
    tail = f"+{plus} -{minus}"
    out.append(dim(f"  … {rest} more lines  ({tail})" if rest
                   else f"  {tail}"))
    return out


# ---------------------------------------------------------------- tool calls
def tool_line(name: str, args: dict, subject_key: str = "") -> str:
    """`bash(pytest -q)` - the same shape the permission prompt and /policy
    use, so the three agree on how a call is named."""
    subject = str(args.get(subject_key, "") or "") if subject_key else ""
    if subject and name == "read_file":
        span = args.get("offset")
        if span:
            subject = f"{subject}:{span}+{args.get('limit', 2000)}"
    if not subject:
        subject = _short(json.dumps(args, ensure_ascii=False), 120)
    return dim(f"  · {name}({_short(subject, 120)})")


def _fit_block(lines: list, room: int, prefer_tail: bool) -> list:
    """At most `room` lines out of the middle of a command's output.

    The "… N more lines …" marker is counted against the budget, so what comes
    back is never longer than what was asked for - the reason to cap it is a
    small terminal, and overshooting there defeats the point.
    """
    if room <= 0 or not lines:
        return []
    if len(lines) <= room:
        return lines
    room -= 1                               # the marker takes a line
    # A failure explains itself at the end; a success is recognised at the top.
    tail = room if prefer_tail else max(0, room - 2)
    head = room - tail
    gap = len(lines) - head - tail
    return lines[:head] + [f"…{gap} more lines…"] + (lines[-tail:] if tail else [])


# t_read's own tail marker, and only that: anchored at the end and shaped like
# `...[N more lines]`.  Matching a bare `...[` anywhere would also hit the one
# _clip leaves in the *middle* of an oversized result, and everything after it
# - thousands of characters of file - would be pasted onto the summary line.
_MORE_LINES = re.compile(r"\n\.\.\.\[\d+ more lines\]\s*\Z")
_CUT_OUT = "chars cut from the middle]"


def tool_result(name: str, result: str, seconds: float = 0.0,
                max_lines: int = 6) -> list:
    """A few dim lines saying what the call actually did.

    Reads the already-clipped string the model was given, so what is on the
    screen can never claim more than what was sent.
    """
    if not max_lines or not result.strip() or result.startswith("DENIED"):
        return []                       # DENIED is printed in red by the caller
    room = max(20, shutil.get_terminal_size().columns - 6)
    took = f"  {seconds:.1f}s" if seconds >= 1.0 else ""

    if result.startswith("ERROR:"):
        return ["    " + yellow(_fit(result.splitlines()[0], room))]

    if name == "bash":
        first, _, rest = result.partition("\n")
        code = first[5:] if first.startswith("exit=") else "?"
        body = [l for l in rest.replace("--- stdout ---", "")
                .replace("--- stderr ---", "").splitlines() if l.strip()]
        head = f"exit {code}  {len(body)} lines{took}"
        out = ["    " + (dim(head) if code == "0" else yellow(head))]
        keep = _fit_block(body, max_lines - 1, code != "0")
        out += ["    " + dim(_fit(l, room)) for l in keep]
        return out

    if name == "read_file":
        m = _MORE_LINES.search(result)
        more = " " + m.group(0).strip() if m else ""
        n = len(result.splitlines()) - (1 if m else 0)
        # A clipped result is missing its middle, so `n` is how much came back
        # rather than how long the file is; say so instead of implying it.
        cut = " (clipped)" if _CUT_OUT in result else ""
        return ["    " + dim(_fit(f"{n} lines{cut}{more}{took}", room))]

    # write_file / edit_file already say +N -M themselves
    return ["    " + dim(_fit(result.splitlines()[0], room) + took)]
