# Model Routing Advisor

A Claude Code skill that advises you when a different Claude model
tier would be more cost-effective for your prompts. It classifies
each prompt's complexity via Haiku and tells you when you're
over- or under-provisioned — while tracking and reporting its own
overhead to stay honest about its value.

## How It Works

1. You activate the skill once per session with
   `/router`.
2. A `UserPromptSubmit` hook registers and fires on every
   subsequent prompt.
3. The hook calls Haiku (~$0.0003 per classification) to
   classify your prompt's complexity into a tier: **haiku**,
   **sonnet**, or **opus**.
4. Claude compares the recommendation against its own model and
   advises you if there's a mismatch.
5. You decide whether to switch models with `/model`.

The skill is **advisory only** — it never changes your model
automatically.

### Self-Deactivation

The router monitors its own value and deactivates when:

- The last 10 recommendations are all the same tier (your model
  is already well-matched).
- The router's cumulative cost exceeds its estimated savings
  (negative ROI).

When it deactivates, it tells you why and reports its total cost.

## Installation

Clone this repository into your Claude Code skills directory:

```bash
cd ~/.claude/skills/
git clone https://github.com/dustinblack/model-routing-skill.git router
```

That's it. No dependencies to install — the hook script uses only
the Python 3.8+ standard library.

**Important:** The skill must be cloned as `router` at
`~/.claude/skills/router/`. The hook command references this
path directly because `${CLAUDE_SKILL_DIR}` substitution is not
supported in hook command fields (only in skill body content and
`allowed-tools`). If you install to a different path, update the
`command` in the `hooks` section of `SKILL.md`.

## Prerequisites

- **Claude Code** with skill frontmatter hooks support
- **Python 3.8+** (used by the hook script, stdlib only)
- **`ANTHROPIC_API_KEY`** environment variable set with a valid
  Anthropic API key (the hook calls Haiku for classification)

## Usage

### Activate routing advice

In any Claude Code session:

```
/router
```

This registers the hook. From this point on, every prompt you
send is classified and you'll receive routing advice when your
current model doesn't match the recommended tier.

### Check routing status

Ask Claude about routing at any time:

```
How is routing going?
```

Claude will report the number of prompts classified, the
router's cumulative cost, and estimated savings.

### What you'll see

**When there's a mismatch** (e.g., you're on Opus but the
prompt is simple):

> I'm running as Opus, but the router suggests Sonnet would
> handle this well — you can switch with `/model` if you'd
> like.

**When you're well-matched:** Nothing — Claude proceeds
normally without mentioning routing.

**When the router deactivates:**

> The model routing advisor has deactivated — all recent
> recommendations have been for Opus, so you're well-matched.
> Router cost this session: $0.001.

## Configuration

Tunable thresholds are in `config/defaults.json`:

| Setting | Default | Description |
|:--------|--------:|:------------|
| `monotonic_threshold` | 10 | Consecutive same-tier recommendations before deactivation |
| `min_prompts_before_deactivation` | 5 | Minimum classifications before deactivation can trigger |
| `max_prompt_chars` | 500 | Prompt text truncated to this length for classification |
| `api_timeout_seconds` | 10 | Haiku API call timeout |
| `haiku_max_tokens` | 150 | Max tokens for the classification response |

## Cost

Each classification call costs approximately **$0.0003**
(~150 input tokens + ~30 output tokens at Haiku pricing). The
router breaks even after redirecting 2-3 prompts from Opus to
Sonnet in a session.

For a session of 50 prompts, total router overhead is
approximately **$0.015**.

See `references/cost-model.md` for detailed pricing data.

## Limitations

- **No cache awareness.** The router cannot observe prompt cache
  state. Switching models invalidates the cache, which may cost
  more than the per-token savings for several turns in long
  conversations. Savings estimates do not account for this.
- **Prompt-only classification.** The router sees only the
  current prompt text, not the full conversation context. A
  prompt like "do the same for the rest" may be classified as
  simple when the actual task is complex. Claude is instructed
  to override the router when it knows better.
- **Baked-in pricing.** Model pricing is hardcoded. Update the
  script or pull the latest version when pricing changes.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
