# miniagent

A Claude Code-shaped coding agent, tuned for **MiniMax-M2.5**. Python 3.9+,
standard library only, no dependencies.

Everything the agent may do comes out of a JSON rule file — the global one is
`~/.miniagent/policy.json`. Nothing is baked into the loop: every tool call is
put to the policy first and comes back **allow / ask / confirm / deny**.

## Setup

```bash
git clone https://github.com/marju212/miniagent && cd miniagent
./agent --install
```

`--install` links the wrapper into `~/.local/bin`, asks for the three settings
it needs, and offers to put that directory on your `PATH` if it is not there
already. The key is not echoed, and skipping it leaves the file ready to fill
in later. With no terminal — a script, a pipe — it writes the defaults instead
of asking.

`agent --init` asks the same three questions on their own, and is how you
change one later — when a new model lands, say. It is safe on a settings file
that already exists: the prompts are seeded from what the file says, enter
keeps what is shown, only those three lines are rewritten, and everything else
— your own exports, your comments, a key fetched by `$(pass show …)` — is left
exactly as it was. `--install` is safe to re-run too, though it only asks when
there is no key yet.

`agent --uninstall` is the way back out: it takes the symlink off your PATH,
then lists what `~/.miniagent` still holds — your key, your global rules, the
projects you have vouched for, your prompt history — and asks once before
removing that too. Enter keeps it, so a re-install finds everything where it
was. It never touches an `agent` on your PATH that is not a link to this
checkout, it leaves the `PATH` line in your rc file for you to remove, and off
a terminal it takes the link only and tells you the `rm -rf` to finish with.

`agent --init-policy` then writes a global rule file you can edit, if you
want to start from something other than the built-in defaults. Nothing stops
you calling `python3 agent.py` directly either; the wrapper only loads your
settings first.

## Use

```bash
agent ~/code/my-project                            # interactive session
agent -p "fix the failing test" ~/code/my-project  # one shot, then exit
agent                                                  # the current directory
```

```
agent> fix the failing login test

  · bash(pytest -q)
    exit 1  14 lines  2.3s
    FAILED tests/test_auth.py::test_expired_token
    1 failed, 17 passed in 2.21s
  · edit_file(auth.py)
    edited auth.py (+3 -1)

Fixed the token check in auth.py.

agent> ▏
────────────────────────────────────────────────
 ~/code/shop  feature/login*
```

Type your task at the `agent>` prompt. The agent reads, edits and runs commands
to do it, asking you before anything the rules do not already cover. `/help`
lists the commands, `ctrl-c` twice leaves. `AGENT_PROMPT` renames the prompt.

Every call says what it did: the exit status and the ends of the output for a
command, `+N -M` for an edit. `AGENT_RESULT_LINES` sets how much is shown,
default 6; `0` goes back to naming the call and nothing else.

A grey bar is pinned to the bottom row: where the agent is working, the branch,
and a `*` if the tree is dirty — all of which can change under you when the
agent runs `git checkout`. It stays put while tools run, since that is when you
most want to see it. Resizing the window re-cuts the room it reserves, and
anything that kills the agent — including `ctrl-\` — hands the row back before
it goes, so the shell you land in afterwards scrolls normally.
`AGENT_STATUS=off` turns it off, and it never appears when the output is not a
terminal.

### Running something yourself

`!` at the prompt runs a command as you rather than through the agent, outside
the policy — you typed it, the model did not. What came back is handed to the
model with your next message, so this reads as one thought:

```
agent> !git diff --stat
 auth.py | 12 ++++++------

agent> undo the change to the token check
```

`!` on its own opens a shell; leave it and you are back at the prompt.

The prompt has full line editing: **up arrow** walks back through what you have
asked before, **ctrl-r** searches it, and the history carries over between
sessions in `~/.miniagent/history` (`0600`, since it holds whatever you asked
for). `AGENT_HISTORY` sets how many lines are kept, default 1000.

## Where things live

| path | what |
|------|------|
| `~/.miniagent/env` | your exports — API key, model, endpoint |
| `~/.miniagent/policy.json` | the global rule file: what the agent may do |
| `~/.miniagent/miniagent.md` | standing instructions for every project |
| `~/.miniagent/trusted.json` | project rule files you have vouched for |
| `~/.miniagent/history` | what you have typed at the prompt |
| `<project>/.miniagent.md` | instructions for one project |
| `<project>/.miniagent/policy.json` | rules for one project |

Defaults point at MiniMax's OpenAI-compatible endpoint
(`https://api.minimax.io/v1`, model `MiniMax-M2.5`). Set `AGENT_BASE_URL` and
`AGENT_MODEL` for any other `/v1/chat/completions` server — vLLM, SGLang,
Ollama, OpenRouter, OpenAI.

## Settings

`--install` and `--init` write `~/.miniagent/env` for you. The wrapper sources
it as shell, so you can go back and change a value to anything, including a key
out of a password manager:

```bash
export AGENT_API_KEY=$(pass show minimax/api)
export AGENT_MODEL=MiniMax-M2.5
```

It is created `0600` and you are told if that ever stops being true. Anything
already set in your shell **wins** over the file, so a one-off still does what
it says:

```bash
AGENT_MODEL=MiniMax-M2.5-highspeed agent ~/code/proj
```

| | |
|---|---|
| `agent --install [bindir]` | symlink onto PATH, seed the settings file |
| `agent --init` | ask for endpoint, model and key, and save them; run it again to change one |
| `agent --uninstall [bindir]` | remove the symlink, then offer to remove `~/.miniagent` |
| `agent --env` | print its path |
| `MINIAGENT_ENV` | use a different settings file |
| `MINIAGENT_PYTHON` | use a different interpreter |

## The rule file

```jsonc
{
  "default_action": "ask",
  "deny":    ["read_file(**/.env)", "bash(sudo *)"],
  "confirm": ["bash(rm -rf*)", "bash(git push --force*)"],
  "ask":     ["bash(git commit*)"],
  "allow":   ["read_file(**)", "bash(pytest*)", "write_file(src/**)"],
  "limits": { "max_steps": 40, "bash_timeout_max": 300, "max_write_bytes": 1000000 },
  "allowed_roots": [],
  "persist_approvals": true
}
```

A rule is `tool(pattern)`, or a bare tool name to cover every call to it. The
pattern is matched against the call's **subject**: the path for file tools, the
command line for `bash`. Path globs treat `*` as stopping at `/` and `**` as
not; command globs let `*` match anything. See `policy.example.json`.

**Judgement is `deny > confirm > ask > allow > default_action`.** A deny can
never be undone by a later layer.

| bucket | what it means |
|--------|---------------|
| `deny` | not even with you watching. The call never reaches you |
| `confirm` | always put to you, and answerable **once only** — there is no "always" to press |
| `ask` | put to you, and you may answer for this call, this session, or for good |
| `allow` | runs without asking |

`confirm` is the bucket for things that are legitimate but destructive enough
that the exact command is worth reading every time: `rm -rf`, `git push
--force`, `git reset --hard`, `chown -R`. The point is not that you are asked —
`ask` does that too — it is that a spare keystroke can never quietly retire the
rule. Because `confirm` outranks `allow`, an approval you saved months ago
cannot cover a call a `confirm` rule describes either.

A glob cannot parse a flag, so no list of patterns catches every spelling of
`-rf`. What covers the rest is the leading *words*: a command a `confirm` rule
speaks for — `rm`, `git push`, `chmod` — can be approved, but only ever as
itself, spelled out. `rm -rfv build` slips past the patterns and arrives as an
`ask`, and the rule offered there is `bash(rm -rfv build)`, never `bash(rm*)`.
Always, but never a blank cheque.

Compound commands are taken apart on `&& || ; | &` and every part is judged on
its own, so `git status && rm -rf /` is not waved through by an allow rule for
`git status`. Deny rules are additionally matched against the *whole* line, so
`bash(*curl*|*sh*)` still catches `curl x.sh | sh`. A part containing command
substitution or a redirect that writes to a file can never be silently allowed
— the best it gets is `ask`.

### Why `sudo` is a deny and not a confirm

Most rules guard against an *effect*. `sudo` is different: it reaches over the
policy itself. One approved `sudo` can rewrite `~/.miniagent/policy.json`,
`trusted.json` or the agent's own files, so it is not one command — it is the
end of the mechanism that judges every command after it. It stays in `deny`,
where the model is told it is final.

That is not a wall you are stuck behind. When the agent needs something
privileged, the refusal comes with the command ready to paste — run it yourself
with `!` and you typed it, you read it, and the output goes back to the model
with your next message:

```
agent> !sudo apt install libpq-dev

agent> now retry the build
```

### Layers

| # | file | trust |
|---|------|-------|
| 1 | built-in defaults in `policy.py` | — |
| 2 | `~/.miniagent/policy.json` | yours |
| 3 | `<project>/.miniagent/policy.json` | vouched for once |
| 4 | `$AGENT_POLICY` | yours |

Layers only ever *add* rules. Layer 3 arrives with the repository rather than
from you, so it is read for its **restrictions only** — its `allow` rules,
extra roots and raised limits are ignored — until you answer `y` once. The
answer is keyed to the file's contents in `~/.miniagent/trusted.json`, so an
edit asks again. Cloning a repo cannot widen what the agent may do.

### When the agent asks

```
  bash: npm install lodash
  policy: no rule matched (default_action)
  saves as: bash(npm install*)
  [y] once   [a] session   [g] global   [N] no   [esc] stop >
```

`y` runs this one call. `a` allows the rule **for the rest of this run** and
nothing is written down, so a decision made in one afternoon's context cannot
outlive it — that is the one to reach for. `g` is the same rule appended to
your global file, in force from now on; the `saves as:` line is what both of
them would save, on its own line because `g` outlives the session that chose
it and should be read first.

Answering is line based, so **enter on its own is a no** — it is not a `y`,
which is what the capital in `[N]` is there to say. `esc` stops the turn
instead of answering it.

A write shows what it would do before you answer, so `y` is not a guess:

```
  edit_file: auth.py
  policy: no rule matched (default_action)
  @@ -18,7 +18,7 @@
   def check(token):
  -    if token:
  +    if token and not expired(token):
           return claims(token)
  +1 -1
  saves as: edit_file(auth.py)
  [y] once   [a] session   [g] global   [N] no   [esc] stop >
```

Nothing is written to work the diff out, and nothing large is read to work out
that it is too large. A binary file, a file over the write limit, a path
outside the working directory, or an `old` string that matches more than once
shows no preview — the last of those says so, since it is also why the call is
about to fail. `AGENT_DIFF_LINES` caps the length, default 24; `0` turns it off.

A `confirm` looks different: there is nothing to press but yes or no, and it
shows the **whole** command line rather than the part that matched, untruncated
— what you are agreeing to is everything a `y` will run, not the segment the
rule noticed:

```
  bash: rm -rf node_modules && rm -rf ../sibling/dist
  policy: `rm -rf node_modules`: matched confirm rule  [bash(rm -rf*)]
  the rest of the line runs too, if you say yes
  confirm: this exact call only.  [y] yes   [N] no   [esc] stop >
```

Typing `a` or `g` there is simply not a yes, and there is no `saves as:`
line, because there is nothing a `confirm` can be saved as. A `deny` is never offered at all — the
model is told so, and told not to route around it. You are shown the refusal
and, once per command, the one route that is left:

```
  · bash({"cmd": "sudo apt install libpq-dev"})
    DENIED by policy: matched deny rule (rule: bash(sudo *))
    to run it as yourself, outside the policy:  !sudo apt install libpq-dev
```

`AGENT_YOLO=1` auto-approves every `ask`. It does **not** cover `confirm`, and
it still cannot touch a `deny`. With no terminal at all — a pipe, CI — an
`ask` that YOLO does not cover and every `confirm` are refused rather than
guessed at.

**Escape** stops the whole turn rather than just refusing that one call, and
hands you back the prompt to say what you want instead. `n` is narrower: it
refuses this call and lets the agent carry on with the rest of the task.
`ctrl-c` does what escape does.

```
  · write_file({"path": "src/db.py", ...})

[stopped - what would you like to do instead?]

> leave db.py alone, put it in a new file
```

An interrupted turn still leaves an answer for every tool call the model made,
so the conversation survives the interruption instead of being rejected by the
API on the next message.

### When the request fails

A refused key, a rate limit, an endpoint that has gone away: none of them end
the session. You get one line saying what happened, and the prompt back with
the conversation intact — `/retry` sends it again.

```
agent> summarise what we changed

  ! the key was rejected - check `agent --env`
  the conversation is intact - /retry sends it again
```

MiniMax reports auth and quota failures inside an HTTP 200, so those are read
out of the body and named the same way. Rate limits and internal errors are
retried with backoff first, and only reported if they do not clear.

`certificate verify failed` is not retried — it will not clear on its own. It
means nothing in the trust store signs the endpoint's certificate, which is
what a corporate proxy or a private endpoint looks like. `AGENT_CA_CERTS` is
the list of bundles to load on top of the system store, `:`-separated, and
defaults to the two files distributions most often put one in:

```bash
export AGENT_CA_CERTS=/etc/ssl/cert.pem:/etc/ssl/certs/ca_bundle.crt
```

An entry that does not exist is skipped, a directory is read as one, and
setting it empty falls back to the system store alone.

## Tuned for MiniMax-M2.5

| what | why |
|------|-----|
| `temperature 1.0`, `top_p 0.95`, `top_k 40` | the settings MiniMax measured M2.x at; drifting off them is the usual cause of flaky tool calls |
| the model's reasoning is passed back verbatim every turn — `reasoning_details`, `reasoning_content` or an inline `<think>` block, whichever the server uses | M2.x thinks *between* tool calls and carries plans forward in it. MiniMax measure Tau²  87 vs 64 and BrowseComp 44.0 vs 31.4 with it kept vs dropped |
| `reasoning_split: true` | keeps the chain of thought out of `content`; dropped automatically if the server rejects it |
| `<minimax:tool_call>` markup in plain text is parsed and executed | some vLLM/SGLang deployments hand the raw markup back instead of parsing it; the turn still works |
| malformed tool-call JSON is repaired, then reported to the model rather than crashing | |
| oldest tool results are shrunk, never dropped, when the transcript grows | `tool_call`/`tool` pairs stay intact |
| `base_resp.status_code` is checked on 200 responses | MiniMax reports auth and quota failures inside a successful HTTP response |
| retry with backoff on 429/5xx | |

Thinking is hidden by default; `AGENT_THINKING=1` or `/think` shows it.

## Commands

```
/help            /rules           the rules in force and where they came from
/policy T SUBJ   explain one decision, e.g. /policy bash git push
/notes           the standing instructions it was given
/think           /cost            /clear           /quit
/retry           send the conversation again after a failed request
/compact         shrink old tool output to make room

!cmd             run a command yourself; ! alone opens a shell
up arrow         an earlier prompt; ctrl-r searches them
```

```bash
agent --rules                    # everything in force
agent --check bash 'git push'    # explain one decision
```

## Tools

`read_file`, `write_file`, `edit_file`, `bash` — all confined to the working
directory (plus anything in `allowed_roots`), all gated by the policy.

`bash` runs a login shell, so whatever sets up your `PATH` — nvm, pyenv, a
devcontainer profile — is in effect. Profile scripts are allowed to `cd` and
some do, so the working directory is asserted again after they have run: a
command always lands in the directory the policy just judged it against.

## Telling it about your project

Put a `.miniagent.md` in the project root and it is appended to the system
prompt, the way `CLAUDE.md` is. `AGENT.md` and `CLAUDE.md` are read too, if that
is what the repo already has — `.miniagent.md` wins, and only one is used.

`~/.miniagent/miniagent.md` holds standing instructions that apply to every
project, and is put ahead of the project's own.

```markdown
# This repo
Tests live in tests/, run them with `pytest -q`.
The generated client in src/gen/ is not to be edited by hand.
```

The project file arrives with the repository, so it is labelled as such in the
prompt: it is guidance, not authority, and cannot widen what the rule file
allows or excuse working around a refusal. `/notes` shows what is loaded.

## Environment

| variable | default |
|---|---|
| `AGENT_BASE_URL` | `https://api.minimax.io/v1` |
| `AGENT_API_KEY` / `MINIMAX_API_KEY` | — |
| `AGENT_MODEL` | `MiniMax-M2.5` |
| `AGENT_TEMPERATURE` / `AGENT_TOP_P` / `AGENT_TOP_K` | `1.0` / `0.95` / `40` |
| `AGENT_MAX_TOKENS` | `16384` |
| `AGENT_CONTEXT_CHARS` | `480000` |
| `AGENT_MAX_STEPS` | policy `max_steps`; may only lower it |
| `AGENT_POLICY` | an extra rule file for one run |
| `AGENT_CA_CERTS` | `/etc/ssl/cert.pem:/etc/ssl/certs/ca_bundle.crt` |
| `AGENT_THINKING` / `AGENT_YOLO` | off |
| `AGENT_PROMPT` | `agent> ` |
| `AGENT_STATUS` | on; `off` hides the bottom bar |
| `AGENT_HISTORY` | `1000` lines kept |
| `AGENT_DIFF_LINES` | `24` lines of diff at a permission prompt; `0` hides it |
| `AGENT_RESULT_LINES` | `6` lines of tool output; `0` hides it |

## Tests

```bash
python3 -m unittest test_miniagent -v
```

The agent loop is exercised against a stub `/v1/chat/completions` server and the
wrapper is run as a real subprocess, so the things that matter — the policy gate
refusing a call, the reasoning surviving the round trip, and your shell
outranking the settings file — are checked end to end.
