#!/usr/bin/env python3
"""
miniagent policy - the JSON file that decides what the agent is allowed to do.

Rules are strings of the form ``tool(pattern)`` (or bare ``tool`` for every
call to it) sorted into three buckets: ``deny``, ``ask`` and ``allow``.
Evaluation is deny-wins:

    deny  >  ask  >  allow  >  default_action

Policies are layered.  Each layer may only *add* rules, so a project file can
never un-deny something the user's own file forbade:

    1. the built-in DEFAULTS below
    2. ~/.miniagent/policy.json          (the global file: the user's own rules)
    3. <root>/.miniagent/policy.json     (per-project, checked into git)
    4. $AGENT_POLICY                     (an explicit file for one run)

Layer 3 arrives with the repository rather than from the user, so it is read
for its *restrictions only* until the user vouches for it once; its ``allow``
rules, extra roots and raised limits are ignored until then.  Cloning a repo
therefore cannot widen what the agent may do behind the user's back.

The pattern is matched against the call's *subject*: the path for file tools,
the command line for ``bash``, "" for anything else.  Path patterns are globs
where ``*`` stops at ``/`` and ``**`` does not; command patterns are globs
where ``*`` matches anything.

Shell commands are split on ``&& || ; | &`` and every segment is judged on its
own, so ``git status && rm -rf /`` is not waved through by an allow rule for
``git status``.  Commands containing substitution or a redirect that writes to
a file can never be silently allowed - the best they get is ``ask``.
"""

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import NamedTuple

ACTIONS = ("deny", "ask", "allow")
_RANK = {"deny": 0, "ask": 1, "allow": 2}  # lower == more restrictive

# Which argument carries the thing we are making a decision about.
SUBJECT = {
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "bash": "cmd",
}

DEFAULTS = {
    "version": 1,
    "default_action": "ask",
    "deny": [
        "read_file(**/.env)",
        "read_file(**/.env.*)",
        "read_file(**/*.pem)",
        "read_file(**/*id_rsa*)",
        "read_file(**/.aws/**)",
        "read_file(**/.ssh/**)",
        "write_file(.git/**)",
        "edit_file(.git/**)",
        "bash(rm -rf /*)",
        "bash(rm -rf ~*)",
        "bash(:(){*)",
        "bash(* > /dev/sd*)",
        "bash(sudo *)",
        "bash(shutdown*)",
        "bash(mkfs*)",
        "bash(dd if=* of=/dev/*)",
        "bash(git push --force*)",
        "bash(git push -f*)",
        "bash(*curl*|*sh*)",
        "bash(*wget*|*sh*)",
    ],
    "ask": [
        "bash(git push*)",
        "bash(git commit*)",
        "bash(gh *)",
        "bash(npm publish*)",
        "bash(pip install*)",
        "bash(npm install*)",
        "bash(curl*)",
        "bash(wget*)",
    ],
    "allow": [
        "read_file(**)",
        "bash(ls*)", "bash(cat *)", "bash(head *)", "bash(tail *)",
        "bash(wc *)", "bash(file *)", "bash(pwd*)", "bash(echo *)",
        "bash(grep *)", "bash(rg *)", "bash(find *)", "bash(which *)",
        "bash(git status*)", "bash(git diff*)", "bash(git log*)",
        "bash(git show*)", "bash(git branch*)", "bash(git add *)",
        "bash(python3 -m pytest*)", "bash(pytest*)",
        "bash(npm test*)", "bash(npm run *)", "bash(make *)",
    ],
    "limits": {
        "max_steps": 40,
        "max_output_chars": 20000,
        "bash_timeout_max": 300,
        "max_write_bytes": 1000000,
    },
    "bash": {
        # Judge every segment of a compound command separately.
        "split_operators": True,
        # $(...), `...`, <(...) and > redirects downgrade allow -> ask.
        "strict_syntax": True,
    },
    # Extra directories the file tools may touch, besides the working dir.
    "allowed_roots": [],
    # May the user turn a one-off approval into a saved rule?
    "persist_approvals": True,
}


class Decision(NamedTuple):
    action: str          # deny | ask | allow
    reason: str          # human-readable, shown at the prompt
    rule: str = ""       # the rule that decided it, if any
    subject: str = ""    # the exact path or command segment it judged

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


# ------------------------------------------------------------------ globs
MAX_PATTERN = 1024


def _any_char(_c: str) -> bool:
    return True


def _not_sep(c: str) -> bool:
    return c != "/"


def _char_class(inner: str):
    """A one-character class.  `re` is safe here - a class cannot backtrack."""
    inner = inner.replace("\\", "\\\\").replace("[", "\\[")   # `[[]` warns about a nested set
    rx = re.compile("[" + inner + "]", re.DOTALL)
    return rx.match


class Glob:
    """A compiled glob pattern.

    Matching is a linear sweep, not a regex search.  Translated to a regex,
    `*x*x*x*xz` costs seconds of backtracking against a few hundred characters,
    and rule files can arrive with a repository - so how a pattern is *written*
    must not decide how long it takes to judge a command.  Cost here is
    len(pattern) x len(subject), and the pattern is bounded.

    With sep_sensitive, `*` stops at `/`, `**` does not, and `**/` spans whole
    path segments.
    """

    __slots__ = ("source", "tokens", "_needs_slashes")

    LIT, ONE, STAR, SEGS = 0, 1, 2, 3   # token kinds

    def __init__(self, pattern: str, sep_sensitive: bool):
        if len(pattern) > MAX_PATTERN:
            raise ValueError(f"pattern is longer than {MAX_PATTERN} characters")
        self.source = pattern
        self.tokens = self._tokenize(pattern, sep_sensitive)
        self._needs_slashes = any(k == self.SEGS for k, _ in self.tokens)

    @classmethod
    def _tokenize(cls, pat: str, sep: bool) -> list:
        out, i = [], 0
        while i < len(pat):
            c = pat[i]
            if c == "*":
                if sep and pat[i:i + 3] == "**/":
                    out.append((cls.SEGS, None))
                    i += 3
                elif pat[i:i + 2] == "**":
                    out.append((cls.STAR, True))
                    i += 2
                else:
                    out.append((cls.STAR, not sep))
                    i += 1
                continue
            if c == "?":
                out.append((cls.ONE, _not_sep if sep else _any_char))
                i += 1
                continue
            if c == "[":
                j = pat.find("]", i + 1)
                if j <= i + 1:  # a lone `[`, or `[]`, which is no class at all
                    out.append((cls.LIT, c))
                    i += 1
                    continue
                inner = pat[i + 1:j]
                inner = ("^" + inner[1:]) if inner.startswith("!") else inner
                out.append((cls.ONE, _char_class(inner)))
                i = j + 1
                continue
            out.append((cls.LIT, c))
            i += 1
        return out

    def match(self, subject: str) -> bool:
        """Does the whole subject match?  One pass per token, no backtracking."""
        n = len(subject)

        # For `**/`: the next `/` at or after each position, n if there is none.
        after = None
        if self._needs_slashes:
            after = [n] * (n + 1)
            for si in range(n - 1, -1, -1):
                after[si] = si if subject[si] == "/" else after[si + 1]

        # row[si] answers "does the rest of the pattern match subject[si:]",
        # so the empty rest of a pattern matches only the empty rest of the
        # subject.  Each token rebuilds the row from the one behind it.
        row = [False] * (n + 1)
        row[n] = True
        for kind, arg in reversed(self.tokens):
            prev, row = row, [False] * (n + 1)
            if kind == Glob.LIT:
                for si in range(n - 1, -1, -1):
                    row[si] = subject[si] == arg and prev[si + 1]
            elif kind == Glob.ONE:
                for si in range(n - 1, -1, -1):
                    row[si] = bool(arg(subject[si])) and prev[si + 1]
            elif kind == Glob.STAR:
                row[n] = prev[n]
                for si in range(n - 1, -1, -1):
                    row[si] = prev[si] or ((arg or subject[si] != "/") and row[si + 1])
            else:  # SEGS: zero or more `[^/]+/`, and the segment end is forced
                row[n] = prev[n]
                for si in range(n - 1, -1, -1):
                    k = after[si]
                    row[si] = prev[si] or (si < k < n and row[k + 1])
        return row[0]


def escape_glob(s: str) -> str:
    """Quote a literal subject so it can be stored as an exact-match rule."""
    # `]` is left alone: outside a class it is already literal, and `[]]` reads
    # as an empty class to `re` rather than as a class holding `]`.
    return re.sub(r"([*?\[])", r"[\1]", s)


# ------------------------------------------------------------------ shell
_OPERATORS = ("&&", "||", ";", "|", "&", "\n")


def split_command(cmd: str) -> list[str]:
    """Split a compound command on shell operators, respecting quotes."""
    segs, buf, i, quote, depth = [], [], 0, "", 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(cmd):
                buf.append(cmd[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < len(cmd):
            buf.append(cmd[i:i + 2])
            i += 2
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        if depth == 0:
            hit = next((op for op in _OPERATORS if cmd.startswith(op, i)), None)
            if hit == "&" and (cmd[i - 1:i] in ("<", ">") or cmd[i + 1:i + 2] == ">"):
                hit = None  # `2>&1`, `>&2`, `&>log`: a file descriptor, not an operator
            if hit:
                segs.append("".join(buf))
                buf = []
                i += len(hit)
                continue
        buf.append(c)
        i += 1
    segs.append("".join(buf))
    cleaned = (s.strip().lstrip("({").strip() for s in segs)
    return [s for s in cleaned if s]


def unsafe_syntax(cmd: str) -> str:
    """Return a reason if the command can reach outside what the rules read."""
    stripped = re.sub(r"'[^']*'", "", cmd)
    if "$(" in stripped or "`" in stripped:
        return "command substitution"
    if "<(" in stripped or ">(" in stripped:
        return "process substitution"
    if re.search(r"(?<![0-9<>])>>?\s*[^&\s]", stripped):
        return "redirect writing to a file"
    return ""


# ------------------------------------------------------------------ policy
class Policy:
    def __init__(self, data: dict, sources: list[str]):
        self.data = data
        self.sources = sources
        self.limits = data.get("limits", {})
        self.bash_opts = data.get("bash", {})
        self.default_action = data.get("default_action", "ask")
        self._compiled = {}
        for action in ACTIONS:
            entries = []
            for r in data.get(action, []):
                try:
                    entries.append((r, *_parse_rule(r)))
                except (ValueError, re.error) as e:
                    raise SystemExit(f"policy: bad {action} rule '{_short(r)}': {e}")
            self._compiled[action] = entries

    # -- matching -----------------------------------------------------
    def _match(self, action: str, tool: str, subject: str) -> str:
        for raw, rtool, rx, sep in self._compiled[action]:
            if rtool != tool:
                continue
            if rx is None or rx.match(subject):
                return raw
        return ""

    def _judge(self, tool: str, subject: str) -> Decision:
        for action in ACTIONS:  # deny, ask, allow
            rule = self._match(action, tool, subject)
            if rule:
                return Decision(action, f"matched {action} rule", rule, subject)
        return Decision(self.default_action, "no rule matched (default_action)",
                        "", subject)

    def check(self, tool: str, args: dict) -> Decision:
        """Decide what to do about one tool call."""
        subject = str(args.get(SUBJECT.get(tool, ""), "") or "")

        if tool != "bash" or not self.bash_opts.get("split_operators", True):
            return self._judge(tool, subject)

        # A deny rule may describe a whole pipeline (`*curl*|*sh*`) that no
        # single segment can match, so judge the full line before splitting it.
        rule = self._match("deny", tool, subject)
        if rule:
            return Decision("deny", "matched deny rule", rule, subject)

        worst = Decision("allow", "no segment objected")
        for seg in split_command(subject) or [subject]:
            d = self._judge(tool, seg)
            if d.action == "allow" and self.bash_opts.get("strict_syntax", True):
                why = unsafe_syntax(seg)
                if why:
                    d = Decision("ask", f"{why} cannot be auto-allowed", d.rule, seg)
            if _RANK[d.action] < _RANK[worst.action]:
                worst = d._replace(reason=f"`{_short(seg)}`: {d.reason}", subject=seg)
            if worst.action == "deny":
                break
        return worst

    # -- paths --------------------------------------------------------
    def roots(self, root: Path) -> list[Path]:
        extra = [Path(os.path.expanduser(p)).resolve()
                 for p in self.data.get("allowed_roots", [])]
        return [root, *extra]

    # -- persistence --------------------------------------------------
    def remember(self, path: Path, rule: str) -> str:
        """Append an allow rule to a policy file and use it from now on."""
        if not self.data.get("persist_approvals", True):
            return "saving approvals is disabled by policy"
        try:
            parsed = _parse_rule(rule)
        except (ValueError, re.error) as e:
            return f"not saved, {rule!r} is not a usable rule: {e}"
        doc = {}
        if path.exists():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                return f"not saved, {path} is not valid JSON: {e}"
        rules = doc.setdefault("allow", [])
        if rule not in rules:
            rules.append(rule)
        doc.setdefault("version", 1)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            return f"not saved: {e}"
        self.data.setdefault("allow", []).append(rule)
        self._compiled["allow"].append((rule, *parsed))
        return f"saved to {path}"


def _short(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _parse_rule(rule: str):
    """'bash(git log*)' -> ('bash', compiled, sep_sensitive)."""
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\)\s*)?$", rule, re.DOTALL)
    if not m:
        raise ValueError(f"malformed policy rule: {rule!r}")
    tool, pat = m.group(1), m.group(2)
    if pat is None:
        return tool, None, False
    sep = tool != "bash"
    return tool, Glob(pat, sep), sep


def _stricter(a: str, b: str) -> str:
    """Whichever of two actions grants less."""
    return a if _RANK.get(a, 1) <= _RANK.get(b, 1) else b


def _merge_limits(base: dict, extra: dict, trusted: bool) -> dict:
    """Numeric limits are ceilings: an untrusted layer may lower, never raise."""
    out = dict(base)
    for k, v in extra.items():
        num = isinstance(v, (int, float)) and isinstance(out.get(k), (int, float))
        out[k] = v if trusted or not num else min(out[k], v)
    return out


def _merge(base: dict, extra: dict, trusted: bool = True) -> dict:
    """Fold one policy layer onto another.

    Rules are only ever added, so a `deny` from below always survives.  A layer
    that is *not* trusted - one that came with the repository rather than from
    the user - is additionally read for its restrictions only: its `allow`
    rules, extra roots and raised limits are dropped.
    """
    out = dict(base)
    for k, v in extra.items():
        if k in ACTIONS and isinstance(v, list):
            if k == "allow" and not trusted:
                continue
            out[k] = list(out.get(k, [])) + [x for x in v if x not in out.get(k, [])]
        elif k == "default_action":
            out[k] = v if trusted else _stricter(out.get(k, "ask"), v)
        elif k == "limits" and isinstance(v, dict):
            out[k] = _merge_limits(out.get(k, {}), v, trusted)
        elif k == "allowed_roots" and isinstance(v, list):
            if trusted:
                out[k] = list(out.get(k, [])) + [x for x in v if x not in out.get(k, [])]
        elif k == "persist_approvals":
            out[k] = bool(v) if trusted else (out.get(k, True) and bool(v))
        elif isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def widens(base: dict, doc: dict) -> bool:
    """Would this layer let the agent do anything it currently may not?"""
    if doc.get("allow") or doc.get("allowed_roots"):
        return True
    want = doc.get("default_action")
    if want and _stricter(base.get("default_action", "ask"), want) != want:
        return True
    have = base.get("limits", {})
    for k, v in (doc.get("limits") or {}).items():
        if isinstance(v, (int, float)) and v > have.get(k, 0):
            return True
    return False


# ------------------------------------------------------------------ trust
TRUST_FILE = Path.home() / ".miniagent" / "trusted.json"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_trust() -> dict:
    try:
        doc = json.loads(TRUST_FILE.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def is_trusted(path: Path, text: str) -> bool:
    """True once the user has vouched for this exact file content."""
    return _read_trust().get(str(path)) == _digest(text)


def trust(path: Path, text: str) -> None:
    """Remember that the user vouched for this exact file content."""
    doc = _read_trust()
    doc[str(path)] = _digest(text)
    TRUST_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRUST_FILE.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ loading
def layers(root: Path) -> list[tuple[Path, bool]]:
    """The files that make up a policy, lowest first, with their trust."""
    out = [
        (Path.home() / ".miniagent" / "policy.json", True),
        (root / ".miniagent" / "policy.json", False),
    ]
    if os.environ.get("AGENT_POLICY"):
        out.append((Path(os.environ["AGENT_POLICY"]).expanduser(), True))
    return out


def load(root: Path, prompt=None) -> Policy:
    """Stack the layers into one Policy.

    `prompt(path, doc) -> bool` is asked once about a project file that wants
    to widen what the agent may do; without it such a file is loaded for its
    restrictions only.
    """
    # a deep copy: `remember` appends to these lists, and DEFAULTS is shared
    data, sources = copy.deepcopy(DEFAULTS), ["built-in defaults"]
    for p, trusted in layers(root):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
            doc = json.loads(text)
        except OSError as e:
            raise SystemExit(f"cannot read policy file {p}: {e}")
        except json.JSONDecodeError as e:
            raise SystemExit(f"policy file {p} is not valid JSON: {e}")
        if not isinstance(doc, dict):
            raise SystemExit(f"policy file {p} must hold a JSON object")
        note = ""
        if not trusted:
            if is_trusted(p, text):
                trusted = True
            elif not widens(data, doc):
                trusted = True  # nothing to vouch for, it only tightens
            elif prompt and prompt(p, doc):
                trust(p, text)
                trusted = True
            else:
                note = "  (untrusted: restrictions only)"
        data = _merge(data, doc, trusted)
        sources.append(f"{p}{note}")
    if data.get("default_action") not in ACTIONS:
        raise SystemExit(f"default_action must be one of {ACTIONS}")
    return Policy(data, sources)


# ------------------------------------------------------------------ cli
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        sys.exit("usage: policy.py <tool> <path-or-command>   # explain a decision")
    pol = load(Path.cwd())
    tool, subject = sys.argv[1], " ".join(sys.argv[2:])
    key = SUBJECT.get(tool, "path")
    d = pol.check(tool, {key: subject})
    print(f"policy sources : {', '.join(pol.sources)}")
    print(f"{tool}({subject})")
    print(f"  -> {d.action.upper()}  ({d.reason}{'; rule: ' + d.rule if d.rule else ''})")
