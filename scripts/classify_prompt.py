#!/usr/bin/env python3
"""Model routing advisor hook script.

Classifies user prompt complexity via the Haiku API and outputs
a routing recommendation as context for Claude Code. Tracks its
own token consumption and deactivates when it determines it is
not providing differential value.

This script is invoked by a UserPromptSubmit hook registered in
the skill's SKILL.md frontmatter. It reads hook input from stdin,
calls the Haiku API, and writes a routing recommendation to
stdout. It always exits 0 to avoid blocking the user's prompt.

Dependencies: Python 3.8+ standard library only.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAIKU_MODEL = "claude-haiku-4-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
VERTEX_API_VERSION = "vertex-2023-10-16"

# Vertex AI endpoint template. Region and project are
# filled at runtime from environment variables or gcloud.
VERTEX_URL_TEMPLATE = (
    "https://{region}-aiplatform.googleapis.com/v1/"
    "projects/{project}/locations/{region}/"
    "publishers/anthropic/models/{model}:rawPredict"
)

# Per-million-token pricing
PRICING = {
    "haiku": {"input": 1.00, "output": 5.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 5.00, "output": 25.00},
}

# Tier ordering for savings calculation
TIER_ORDER = ["haiku", "sonnet", "opus"]

# Assumed average tokens per prompt for savings estimation
AVG_INPUT_TOKENS = 1000
AVG_OUTPUT_TOKENS = 500

# Default config values (overridden by config/defaults.json)
DEFAULTS = {
    "monotonic_threshold": 10,
    "min_prompts_before_deactivation": 5,
    "max_prompt_chars": 500,
    "api_timeout_seconds": 10,
    "haiku_max_tokens": 150,
}

# Classification prompt template — kept minimal to reduce cost
CLASSIFICATION_PROMPT = """\
Classify this prompt's complexity for an AI coding assistant.
Reply with exactly one JSON object, no other text.
{{"tier":"haiku|sonnet|opus","reason":"<10 words>"}}

Tiers:
- haiku: Simple lookups, formatting, short explanations, trivial edits
- sonnet: Moderate coding, debugging, refactoring, code review, \
multi-file changes
- opus: Complex architecture, large-scale refactoring, deep research, \
novel algorithms, multi-step planning

Note: Short or ambiguous prompts that reference prior conversation \
context (e.g. "do it", "same for the rest") should default to the \
higher tier since the actual task complexity is unknown.

Prompt: {prompt}"""

# Prefix for all router output messages
OUTPUT_PREFIX = "[Model Router]"

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("model-router")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config():
    """Load configuration from defaults.json, falling back to
    hardcoded defaults.

    Returns:
        dict: Merged configuration values.
    """
    config = dict(DEFAULTS)
    skill_dir = os.environ.get("CLAUDE_SKILL_DIR", "")
    config_path = os.path.join(skill_dir, "config", "defaults.json")
    if skill_dir and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                user_config = json.load(fh)
            config.update(user_config)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read config %s: %s", config_path, exc
            )
    return config


# ---------------------------------------------------------------------------
# Session state management
# ---------------------------------------------------------------------------


def state_path(session_id):
    """Return the path to the session state file.

    Args:
        session_id: The Claude Code session identifier.

    Returns:
        str: Absolute path to the state JSON file in the
             system temp directory.
    """
    return os.path.join(
        tempfile.gettempdir(),
        f"model-router-{session_id}.json",
    )


def load_state(path):
    """Load session state from disk or return fresh defaults.

    Args:
        path: Absolute path to the state JSON file.

    Returns:
        dict: Session state with all required keys guaranteed.
    """
    defaults = {
        "prompt_count": 0,
        "classifications": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "router_cost_usd": 0.0,
        "estimated_savings_usd": 0.0,
        "deactivated": False,
        "deactivation_reason": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            defaults.update(stored)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read state %s: %s", path, exc)
    return defaults


def save_state(path, state):
    """Atomically write session state to disk.

    Writes to a temporary file first, then renames to avoid
    corruption from interrupted writes.

    Args:
        path: Absolute path to the state JSON file.
        state: dict of session state to persist.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning("Failed to save state %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Haiku API classification
# ---------------------------------------------------------------------------


def get_vertex_access_token():
    """Obtain a GCP access token via gcloud CLI.

    Returns:
        str: Bearer token for Vertex AI requests.

    Raises:
        RuntimeError: If gcloud is not installed or fails.
    """
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gcloud auth failed: {result.stderr.strip()}"
            )
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError(
            "gcloud CLI not found. Install the Google Cloud "
            "SDK or set ANTHROPIC_API_KEY for direct API "
            "access."
        )


def get_vertex_config():
    """Resolve Vertex AI project and region from environment
    or gcloud config.

    Checks environment variables first, then falls back to
    gcloud config values.

    Returns:
        tuple: (project_id, region) strings.

    Raises:
        RuntimeError: If project or region cannot be resolved.
    """
    project = os.environ.get(
        "ANTHROPIC_VERTEX_PROJECT_ID",
        os.environ.get(
            "GOOGLE_CLOUD_PROJECT",
            os.environ.get("CLOUDSDK_CORE_PROJECT", ""),
        ),
    )
    region = os.environ.get(
        "CLOUD_ML_REGION",
        os.environ.get("GOOGLE_CLOUD_REGION", ""),
    )

    # Fall back to gcloud config if env vars are not set
    if not project:
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            project = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not region:
        try:
            result = subprocess.run(
                [
                    "gcloud", "config", "get-value",
                    "compute/region",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            region = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not project:
        raise RuntimeError(
            "GCP project not found. Set "
            "ANTHROPIC_VERTEX_PROJECT_ID, "
            "GOOGLE_CLOUD_PROJECT, or configure gcloud."
        )
    if not region:
        raise RuntimeError(
            "GCP region not found. Set CLOUD_ML_REGION, "
            "GOOGLE_CLOUD_REGION, or configure gcloud "
            "compute/region."
        )
    return project, region


def resolve_api_backend():
    """Determine which API backend to use for Haiku calls.

    Checks for ANTHROPIC_API_KEY first (direct API), then
    falls back to Vertex AI credentials.

    Returns:
        tuple: (backend, credentials) where backend is
               "anthropic" or "vertex", and credentials is
               the API key string or a dict with project,
               region, and access_token.

    Raises:
        RuntimeError: If no usable credentials are found.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        return "anthropic", api_key

    # Try Vertex AI
    try:
        project, region = get_vertex_config()
        token = get_vertex_access_token()
        return "vertex", {
            "project": project,
            "region": region,
            "access_token": token,
        }
    except RuntimeError:
        pass

    raise RuntimeError(
        "No API credentials found. Set ANTHROPIC_API_KEY "
        "for direct access, or configure gcloud for "
        "Vertex AI."
    )


def call_haiku_anthropic(api_key, prompt_text, config):
    """Call Haiku via the direct Anthropic API.

    Args:
        api_key: Anthropic API key string.
        prompt_text: The user's prompt, already truncated.
        config: Configuration dict.

    Returns:
        tuple: (tier, reason, input_tokens, output_tokens).
    """
    classification_input = CLASSIFICATION_PROMPT.format(
        prompt=prompt_text
    )

    request_body = json.dumps({
        "model": HAIKU_MODEL,
        "max_tokens": config["haiku_max_tokens"],
        "messages": [
            {"role": "user", "content": classification_input}
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
        },
        method="POST",
    )

    timeout = config["api_timeout_seconds"]
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response_data = json.loads(
            resp.read().decode("utf-8")
        )

    return _parse_haiku_response(response_data)


def call_haiku_vertex(creds, prompt_text, config):
    """Call Haiku via Google Vertex AI rawPredict.

    The Vertex AI endpoint uses Bearer token auth and a
    slightly different request body format: model is in the
    URL path (not the body), and anthropic_version goes in
    the body instead of a header.

    Args:
        creds: Dict with project, region, and access_token.
        prompt_text: The user's prompt, already truncated.
        config: Configuration dict.

    Returns:
        tuple: (tier, reason, input_tokens, output_tokens).
    """
    classification_input = CLASSIFICATION_PROMPT.format(
        prompt=prompt_text
    )

    url = VERTEX_URL_TEMPLATE.format(
        region=creds["region"],
        project=creds["project"],
        model=HAIKU_MODEL,
    )

    # Vertex rawPredict body: no model field, uses
    # anthropic_version instead of the header
    request_body = json.dumps({
        "anthropic_version": VERTEX_API_VERSION,
        "max_tokens": config["haiku_max_tokens"],
        "messages": [
            {"role": "user", "content": classification_input}
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": (
                f"Bearer {creds['access_token']}"
            ),
        },
        method="POST",
    )

    timeout = config["api_timeout_seconds"]
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response_data = json.loads(
            resp.read().decode("utf-8")
        )

    return _parse_haiku_response(response_data)


def call_haiku(backend, creds, prompt_text, config):
    """Call Haiku to classify prompt complexity.

    Dispatches to the appropriate backend (direct Anthropic
    API or Vertex AI) based on the resolved credentials.

    Args:
        backend: "anthropic" or "vertex".
        creds: API key string (anthropic) or dict with
               project, region, access_token (vertex).
        prompt_text: The user's prompt, already truncated.
        config: Configuration dict.

    Returns:
        tuple: (tier, reason, input_tokens, output_tokens).
    """
    if backend == "vertex":
        return call_haiku_vertex(creds, prompt_text, config)
    return call_haiku_anthropic(creds, prompt_text, config)


def _parse_haiku_response(response_data):
    """Extract classification from a Haiku API response.

    Works for both direct Anthropic and Vertex AI responses
    since the response format is identical.

    Args:
        response_data: Parsed JSON response dict.

    Returns:
        tuple: (tier, reason, input_tokens, output_tokens).
    """
    usage = response_data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    content_blocks = response_data.get("content", [])
    response_text = ""
    for block in content_blocks:
        if block.get("type") == "text":
            response_text += block.get("text", "")

    tier, reason = parse_classification(response_text)
    return tier, reason, input_tokens, output_tokens


def parse_classification(text):
    """Parse the tier and reason from Haiku's response text.

    Attempts JSON parsing first, then falls back to regex
    extraction.

    Args:
        text: Raw text response from Haiku.

    Returns:
        tuple: (tier, reason) where tier is one of
               "haiku", "sonnet", "opus" and reason is a
               short explanation string.

    Raises:
        ValueError: If no valid tier can be extracted.
    """
    valid_tiers = {"haiku", "sonnet", "opus"}

    # Try JSON parse first
    try:
        data = json.loads(text.strip())
        tier = data.get("tier", "").lower().strip()
        reason = data.get("reason", "unknown")
        if tier in valid_tiers:
            return tier, reason
    except (json.JSONDecodeError, AttributeError):
        pass

    # Regex fallback: look for tier name in quotes
    match = re.search(
        r'"tier"\s*:\s*"(haiku|sonnet|opus)"',
        text,
        re.IGNORECASE,
    )
    if match:
        tier = match.group(1).lower()
        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]*)"', text
        )
        reason = reason_match.group(1) if reason_match else "unknown"
        return tier, reason

    raise ValueError(f"Could not parse classification from: {text}")


# ---------------------------------------------------------------------------
# Cost and savings calculations
# ---------------------------------------------------------------------------


def calculate_cost(input_tokens, output_tokens):
    """Calculate the dollar cost of a Haiku API call.

    Args:
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens consumed.

    Returns:
        float: Estimated cost in USD.
    """
    haiku = PRICING["haiku"]
    return (
        (input_tokens * haiku["input"] / 1_000_000)
        + (output_tokens * haiku["output"] / 1_000_000)
    )


def estimate_prompt_savings(recommended_tier):
    """Estimate the savings for a single prompt if the user
    switches from one tier above to the recommended tier.

    For "opus" recommendations there are no savings (already the
    highest tier). For "sonnet" we compare against opus pricing.
    For "haiku" we compare against sonnet pricing.

    Args:
        recommended_tier: The tier recommended by the classifier.

    Returns:
        float: Estimated savings in USD for one prompt.
    """
    tier_idx = TIER_ORDER.index(recommended_tier)
    if tier_idx >= len(TIER_ORDER) - 1:
        # Already the highest tier — no savings possible
        return 0.0

    higher_tier = TIER_ORDER[tier_idx + 1]
    higher = PRICING[higher_tier]
    lower = PRICING[recommended_tier]

    savings_input = (
        (higher["input"] - lower["input"])
        * AVG_INPUT_TOKENS
        / 1_000_000
    )
    savings_output = (
        (higher["output"] - lower["output"])
        * AVG_OUTPUT_TOKENS
        / 1_000_000
    )
    return savings_input + savings_output


# ---------------------------------------------------------------------------
# Self-deactivation logic
# ---------------------------------------------------------------------------


def check_deactivation(state, config):
    """Determine whether the router should deactivate itself.

    Two conditions are checked (neither fires until at least
    min_prompts_before_deactivation classifications have been
    made):

    1. Monotonic recommendations — the last N classifications
       are all the same tier.
    2. Negative ROI — cumulative router cost exceeds cumulative
       estimated savings.

    Args:
        state: Current session state dict.
        config: Configuration dict with threshold values.

    Returns:
        tuple: (should_deactivate, reason) where reason is a
               human-readable explanation or None.
    """
    min_prompts = config["min_prompts_before_deactivation"]
    threshold = config["monotonic_threshold"]

    if state["prompt_count"] < min_prompts:
        return False, None

    # Condition 1: monotonic recommendations
    recent = state["classifications"][-threshold:]
    if (
        len(recent) >= threshold
        and len(set(recent)) == 1
    ):
        tier = recent[0]
        return True, (
            f"All last {threshold} recommendations were "
            f"'{tier}' -- routing is not providing "
            f"differential value."
        )

    # Condition 2: negative ROI
    if (
        state["router_cost_usd"] > 0
        and state["estimated_savings_usd"] > 0
        and state["router_cost_usd"]
        > state["estimated_savings_usd"]
    ):
        cost = state["router_cost_usd"]
        savings = state["estimated_savings_usd"]
        return True, (
            f"Router cost (${cost:.4f}) exceeds estimated "
            f"savings (${savings:.4f}) -- negative ROI."
        )

    return False, None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_recommendation(tier, reason, state):
    """Format a normal routing recommendation message.

    Args:
        tier: Recommended model tier string.
        reason: Short reason from the classifier.
        state: Current session state dict.

    Returns:
        str: Formatted recommendation message.
    """
    cost = state["router_cost_usd"]
    savings = state["estimated_savings_usd"]
    count = state["prompt_count"]
    return (
        f"{OUTPUT_PREFIX} Recommended: {tier} "
        f"({reason}) | Session: {count} classified, "
        f"router cost: ${cost:.4f}, "
        f"est. savings: ${savings:.4f}"
    )


def format_deactivation(reason, state):
    """Format a deactivation notification message.

    Args:
        reason: Human-readable deactivation reason.
        state: Current session state dict.

    Returns:
        str: Formatted deactivation message.
    """
    cost = state["router_cost_usd"]
    count = state["prompt_count"]
    return (
        f"{OUTPUT_PREFIX} DEACTIVATED: {reason} "
        f"Session total: {count} classified, "
        f"router cost: ${cost:.4f}"
    )


def format_inactive():
    """Format a message for when the router is already deactivated.

    Returns:
        str: Short inactive status message.
    """
    return f"{OUTPUT_PREFIX} Inactive (previously deactivated)."


def format_error(message):
    """Format an error message.

    Args:
        message: Description of what went wrong.

    Returns:
        str: Formatted error message.
    """
    return (
        f"{OUTPUT_PREFIX} Classification unavailable "
        f"({message}). Prompt not classified."
    )


def output_message(message):
    """Write the routing message to stdout for Claude to see.

    For UserPromptSubmit hooks, plain-text stdout is added as
    context visible to Claude. We output plain text rather than
    JSON since that is sufficient and simpler.

    Args:
        message: The message string to output.
    """
    print(message)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """Main hook entry point.

    Reads UserPromptSubmit hook input from stdin, classifies
    the prompt via Haiku, updates session state, checks
    deactivation conditions, and outputs a routing
    recommendation. Always exits 0.
    """
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        output_message(format_error(f"invalid hook input: {exc}"))
        return

    session_id = hook_input.get("session_id", "unknown")
    user_prompt = hook_input.get("user_prompt", "")

    # Resolve API credentials (direct Anthropic or Vertex AI)
    try:
        backend, creds = resolve_api_backend()
    except RuntimeError as exc:
        output_message(
            f"{OUTPUT_PREFIX} {exc}"
        )
        return

    # Load config and state
    config = load_config()
    spath = state_path(session_id)
    state = load_state(spath)

    # If already deactivated, output minimal message and return
    if state["deactivated"]:
        output_message(format_inactive())
        return

    # Truncate prompt for classification
    max_chars = config["max_prompt_chars"]
    truncated = user_prompt[:max_chars]
    if len(user_prompt) > max_chars:
        truncated += "..."

    # Call Haiku for classification
    try:
        tier, reason, in_tok, out_tok = call_haiku(
            backend, creds, truncated, config
        )
    except urllib.error.URLError as exc:
        output_message(format_error(f"network error: {exc.reason}"))
        save_state(spath, state)
        return
    except Exception as exc:
        output_message(format_error(str(exc)))
        save_state(spath, state)
        return

    # Update state
    state["prompt_count"] += 1
    state["classifications"].append(tier)
    # Keep only last 20 classifications to bound state size
    if len(state["classifications"]) > 20:
        state["classifications"] = state["classifications"][-20:]

    state["total_input_tokens"] += in_tok
    state["total_output_tokens"] += out_tok

    call_cost = calculate_cost(in_tok, out_tok)
    state["router_cost_usd"] += call_cost

    prompt_savings = estimate_prompt_savings(tier)
    state["estimated_savings_usd"] += prompt_savings

    # Check deactivation
    should_deactivate, deact_reason = check_deactivation(
        state, config
    )

    if should_deactivate:
        state["deactivated"] = True
        state["deactivation_reason"] = deact_reason
        save_state(spath, state)
        output_message(format_deactivation(deact_reason, state))
        return

    # Normal output
    save_state(spath, state)
    output_message(format_recommendation(tier, reason, state))


if __name__ == "__main__":
    main()
