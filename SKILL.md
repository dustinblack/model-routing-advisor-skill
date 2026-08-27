---
name: router
description: >
  Classifies prompt complexity and advises when a different Claude
  model tier would be more cost-effective. Invoke to activate
  per-prompt monitoring for the rest of the session. Use when the
  user asks to optimize model costs, enable routing advice, or
  wants cost-performance guidance.
disable-model-invocation: true
hooks:
  UserPromptSubmit:
    - hooks:
        - type: command
          command: "python3 ${CLAUDE_SKILL_DIR}/scripts/classify_prompt.py"
          timeout: 15
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Model Routing Advisor

You have activated per-prompt model routing advice. A background
hook now classifies every user prompt via Haiku and injects a
`[Model Router]` message into your context. Follow these rules
to interpret and relay that information.

## Interpreting routing messages

Each `[Model Router]` message contains a recommended tier
(haiku, sonnet, or opus) and session statistics. Compare the
recommended tier against **your own model identity** — you
know which model you are.

- **Match** (recommendation equals your tier): Say nothing
  about routing. Proceed with the task normally.
- **Mismatch** (recommendation differs from your tier):
  Mention it in one sentence at the start of your response,
  then proceed with the task. Example: "I'm running as Opus,
  but the router suggests Sonnet would handle this well — you
  can switch with `/model` if you'd like."
- **Deactivation**: The router may deactivate itself when it
  determines it is not providing value. Relay the deactivation
  message once, including the reason and session cost.
- **Error**: If the router reports a classification error,
  mention it briefly ("routing unavailable this turn") and
  proceed.

## When to surface statistics

The `[Model Router]` messages include session stats (prompts
classified, router cost, estimated savings). Only surface
these when:

- The user explicitly asks about routing, cost, or stats.
- The router deactivates (include final stats).
- You judge the user would benefit from knowing (e.g., they
  mentioned being cost-conscious).

Do not include stats in every response — keep routing advice
minimal and non-intrusive.

## Use your own judgment

The router classifies based on the prompt text alone. It
cannot see the full conversation context. If a prompt looks
simple (e.g., "do that for the other files") but you know from
context that the task is complex, override the router's
recommendation silently. Your understanding of the task always
takes precedence.

## Pricing reference

For detailed model pricing, see
[cost-model.md](references/cost-model.md).
