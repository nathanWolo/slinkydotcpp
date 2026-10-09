# Battlesnake reference docs

This folder is a local reference for the game rules and the HTTP API.

| File | What it covers |
|---|---|
| [`ENGINE_RULES.md`](ENGINE_RULES.md) | **Start here when implementing a simulator.** Exact turn resolution, elimination order, the tail rule, and food and hazard spawning. Taken from the official rules engine source. |
| [`official/rules.md`](official/rules.md) | Official game rules (prose overview). |
| [`official/api/`](official/api/) | Official API reference: endpoints (`webhooks`), request and response objects (`objects/`), and an example `/move` request. |
| [`official/maps/`](official/maps/) | Standard, Royale, Constrictor and Snail Mode maps. |
| [`official/guides/`](official/guides/) | Competitive play tips and useful algorithms. |
| [`official/faq.md`](official/faq.md) | FAQ. |

## Provenance

- **`official/`** is copied from
  <https://github.com/BattlesnakeOfficial/docs> at commit `dfa3460`
  (2026-09-25). These are the sources of <https://docs.battlesnake.com>.
  They are MIT licensed; see `official/LICENSE`.
  - Some Docusaurus markup (`:::tip`, `<img src={require(...)}>`) is left
    as-is.
  - Links like `/rules` or `/api` point to pages on docs.battlesnake.com.
- **`ENGINE_RULES.md`** was written from reading
  <https://github.com/BattlesnakeOfficial/rules> at commit `87e094e`. That
  engine is AGPL-3.0, so only a description of its behaviour is included
  here, not its code.
  - To read the engine source, or to run local games with the `battlesnake`
    CLI, clone it separately:
    `git clone https://github.com/BattlesnakeOfficial/rules`
