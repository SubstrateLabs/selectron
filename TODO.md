- stop button for selection (running -> stop)
- parser generation workflow
- after generation, start parsing
- parser fallback should be better 
- parser saved in app dir if you're not in repo
  - registry fallback
- button to delete parser?
- save results to duckdb (app dir)
- sdk? 
  - parse fn (done)
  - eval fn
  - test on example code
- pipx install?

## polish
- organize prompts, always use dedent?
- allow configuring model in tui?

## publish
- readme
- CI

## issues
- sometimes selector agent gets "lost in the sauce" with tool attempts
- many tool calls in selector proposal can result in exceeding context length
  - trimmed down tool results so might be better (see selector_tools constants to tune)
- allow multiple parsers to be registered per url?

