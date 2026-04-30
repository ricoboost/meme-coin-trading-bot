# Contributing

This is a personal research project published for educational value. It is
not actively developed and there is no guaranteed review SLA on issues or
pull requests.

That said, the project benefits from outside eyes. If you want to contribute:

## Issues

Useful issue topics:

- Bug reports with a minimal reproduction (env, command, expected vs.
  actual, paper or live mode).
- Documentation gaps or errors — what's missing, what's wrong.
- Security observations (private disclosure preferred — see SECURITY).
- Strategy critiques backed by data (parquet/CSV from your run, please).

Less useful:

- "It doesn't work" without context.
- Requests for live-trading support or guarantees.
- "Will this make money?" — see [DISCLAIMER.md](./DISCLAIMER.md).

## Pull requests

Before opening a PR:

1. Open an issue first if the change is non-trivial. Saves both of us time
   if the direction isn't going to land.
2. Keep the diff small and focused. One logical change per PR.
3. Run `python -m compileall src` before pushing — at minimum.
4. If you change strategy/portfolio/execution code, explain *why* in the
   PR body. "I think this is better" is not enough; share what you ran,
   what you measured.

## Style

- No reformatting passes that aren't strictly needed for the change.
- Match the surrounding style (this codebase mixes a few conventions —
  follow whatever's local).
- New code without tests is fine; new public APIs without docstrings is not.

## What this project will not accept

- Adding paid-only data sources or services as required dependencies.
- Adding analytics, telemetry, or "phone home" behavior.
- Bundling private vendor APIs or scraping tools whose terms forbid it.
  (The collectors here are templates — see [docs/COLLECTORS.md](./docs/COLLECTORS.md).)
- Marketing, "blog post" style README rewrites, or affiliate links.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](./LICENSE) license.
