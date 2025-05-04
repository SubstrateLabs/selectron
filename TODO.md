- function without AI?
- sdk? 
  - parse(url)
  - test parse on a single example in script
  - remove old codegen script
- publish
  - test pipx usage
  - test sdk usage

## polish
- reuse duckdb connection
- better model config options?
- organize prompts, always use dedent
- CI tests

## publish
- readme

## issues
- sometimes selector agent gets "lost in the sauce" with tool attempts
- parser codegen could be higher quality (twitter followers example)
  - better process for iterating on codegen parsers? (e.g. using IDE)
- many tool calls in selector proposal can result in exceeding context length
  - trimmed down tool results so might be better (see selector_tools constants to tune)
- allow multiple parsers to be registered per url?

