# miniagent

A Claude Code-shaped coding agent, tuned for **MiniMax-M2.5**. Python 3.9+,
standard library only, no dependencies.

Everything the agent may do comes out of a JSON rule file — the global one is
`~/.miniagent/policy.json`. Nothing is baked into the loop: every tool call is
put to the policy first and comes back **allow / ask / deny**.

## Setup

```bash
git clone https://github.com/marju212/miniagent && cd miniagent
./miniagent --install
```

```
linked  ~/.local/bin/miniagent -> ~/code/miniagent/miniagent

settings -> ~/.miniagent/env
  press enter to take the default
  endpoint [https://api.minimax.io/v1]: ▏
  model [MiniMax-M2.5]:
  api key (hidden; enter to fill in later):
wrote   ~/.miniagent/env
ready   ~/.local/bin is on your PATH
```

`--install` links the wrapper into `~/.local/bin`, asks for the three settings
it needs, and offers to put that directory on your `PATH` if it is not there
already. Anything already exported is taken as the default, the key is not
echoed, and skipping it leaves the file ready to fill in later. With no
terminal — a script, a pipe — it writes the defaults instead of asking.

`miniagent --init-policy` then writes a global rule file you can edit, if you
want to start from something other than the built-in defaults. Nothing stops
you calling `python3 agent.py` directly either; the wrapper only loads your
settings first.

## Use

```bash
miniagent ~/code/my-project                            # interactive session
miniagent -p "fix the failing test" ~/code/my-project  # one shot, then exit
miniagent                                              # the current directory
```

```
agent> fix the failing login test

  · bash({"cmd": "pytest -q"})
  · edit_file({"path": "auth.py"})

Fixed the token check in auth.py.

agent> ▏
────────────────────────────────────────────────
 ~/code/shop  feature/login*
```

Type your task at the `agent>` prompt. The agent reads, edits and runs commands
to do it, asking you before anything the rules do not already cover. `/help`
lists the commands, `ctrl-c` leaves. `AGENT_PROMPT` renames the prompt.

A grey bar is pinned to the bottom row: where the agent is working, the branch,
and a `*` if the tree is dirty — all of which can change under you when the
agent runs `git checkout`. It stays put while tools run, since that is when you
most want to see it. `AGENT_STATUS=off` turns it off, and it never appears when
the output is not a terminal.

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
AGENT_MODEL=MiniMax-M2.5-highspeed miniagent ~/code/proj
```

| | |
|---|---|
| `miniagent --install [bindir]` | symlink onto PATH, seed the settings file |
| `miniagent --init` | ask for endpoint, model and key, and save them |
| `miniagent --env` | print its path |
| `MINIAGENT_ENV` | use a different settings file |
| `MINIAGENT_PYTHON` | use a different interpreter |

## The rule file

```jsonc
{
  "default_action": "ask",
  "deny":  ["read_file(**/.env)", "bash(git push --force*)"],
  "ask":   ["bash(git commit*)"],
  "allow": ["read_file(**)", "bash(pytest*)", "write_file(src/**)"],
  "limits": { "max_steps": 40, "bash_timeout_max": 300, "max_write_bytes": 1000000 },
  "allowed_roots": [],
  "persist_approvals": true
}
```

A rule is `tool(pattern)`, or a bare tool name to cover every call to it. The
pattern is matched against the call's **subject**: the path for file tools, the
command line for `bash`. Path globs treat `*` as stopping at `/` and `**` as
not; command globs let `*` match anything. See `policy.example.json`.

**Judgement is `deny > ask > allow > default_action`.** A deny can never be
undone by a later layer.

Compound commands are taken apart on `&& || ; | &` and every part is judged on
its own, so `git status && rm -rf /` is not waved through by an allow rule for
`git status`. Deny rules are additionally matched against the *whole* line, so
`bash(*curl*|*sh*)` still catches `curl x.sh | sh`. A part containing command
substitution or a redirect that writes to a file can never be silently allowed
— the best it gets is `ask`.

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
  [y] once   [a] always, save bash(npm install*)   [N] no   [esc] stop >
```

`a` appends the rule to your global file and it takes effect immediately. A
`deny` is never offered — the model is told so, and told not to route around
it. `AGENT_YOLO=1` auto-approves every `ask`; it still cannot touch a `deny`.

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
/think           /cost            /reset           /quit

!cmd             run a command yourself; ! alone opens a shell
up arrow         an earlier prompt; ctrl-r searches them
```

```bash
miniagent --rules                    # everything in force
miniagent --check bash 'git push'    # explain one decision
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
| `AGENT_THINKING` / `AGENT_YOLO` | off |
| `AGENT_PROMPT` | `agent> ` |
| `AGENT_STATUS` | on; `off` hides the bottom bar |
| `AGENT_HISTORY` | `1000` lines kept |

## Tests

```bash
python3 -m unittest test_miniagent -v
```

The agent loop is exercised against a stub `/v1/chat/completions` server and the
wrapper is run as a real subprocess, so the things that matter — the policy gate
refusing a call, the reasoning surviving the round trip, and your shell
outranking the settings file — are checked end to end.
