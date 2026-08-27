# Claude Model Pricing Reference

Use this reference when interpreting routing recommendations and
estimating cost savings.

## Per-Token Pricing (per 1M tokens)

| Model      | Input  | Output  | Cache Write | Cache Read |
|:-----------|-------:|--------:|------------:|-----------:|
| Haiku 4.5  |  $1.00 |   $5.00 |       $1.25 |      $0.10 |
| Sonnet 4.6 |  $3.00 |  $15.00 |       $3.75 |      $0.30 |
| Opus 4.8   |  $5.00 |  $25.00 |       $6.25 |      $0.50 |

## Tier Savings per Prompt (estimated)

Assumes an average prompt of ~1,000 input tokens and ~500 output
tokens at full price (no cache). Actual savings depend on prompt
size and cache state, which the router cannot observe.

| Switch              | Est. savings per prompt |
|:--------------------|------------------------:|
| Opus -> Sonnet      |                  $0.007 |
| Opus -> Haiku       |                  $0.014 |
| Sonnet -> Haiku     |                  $0.007 |

## Router Overhead

Each classification call to Haiku costs approximately $0.0003
(~150 input tokens + ~30 output tokens). The router breaks even
after redirecting 2-3 prompts from Opus to Sonnet.

## Cache Considerations

Switching models invalidates the prompt cache — each model has its
own cache. A switch to a cheaper model means one full-price
(uncached) turn while the new model warms its cache. For long
conversations, the cache rebuild cost may exceed the per-token
savings for several turns. The router's savings estimates do not
account for cache effects.
