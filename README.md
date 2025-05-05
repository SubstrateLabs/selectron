# ⣏ Selectron ⣹

[![PyPI - Version](https://img.shields.io/pypi/v/selectron.svg)](https://pypi.org/project/selectron)

Selectron is an AI web parsing library & CLI designed around two goals:
1. **Fully automated parser generation** ("AI-compiled" code) 
2. **Efficient parser execution** (No AI at runtime)

![screenshot](/app.png)

<details> 
<summary><h3>Demo videos</h3></summary>

<h4>Save your Twitter feed to DuckDB</h4>

https://github.com/user-attachments/assets/d8743c32-087f-4137-8451-e4ec3e5716ed

<h4>Generate a new scraper with AI</h4>

https://github.com/user-attachments/assets/8f523f33-a786-4871-b081-4fe9b7422a44

</details>

## How it works

- **Chrome integration:** Connects to Chrome over CDP and receives live DOM and screenshot data from your active tab. Selectron uses minimal [dependencies](https://github.com/SubstrateLabs/selectron/blob/main/pyproject.toml) – no [browser-use](https://github.com/browser-use/browser-use) or [stagehand](https://github.com/browserbase/stagehand), not even Playwright (we prefer [direct CDP](https://github.com/SubstrateLabs/selectron/blob/main/src/selectron/chrome/chrome_cdp.py)).
- **Fully automated parser generation:** An AI agent generates selectors for content described with natural language. Another agent generates code to extract data from selected containers. The final result is a serialized [parser](https://github.com/SubstrateLabs/selectron/blob/main/src/selectron/parsers/news.ycombinator.com.json). 
- **CLI application:** When you run the [Textual](https://www.textualize.io) CLI, parsed data is saved to a [DuckDB](https://duckdb.org) database, making it easy to analyze your browsing history or extract structured data from websites. Built-in parsers include:
   - **Twitter**
   - **LinkedIn**
   - **HackerNews**
   - (Please [contribute](https://github.com/SubstrateLabs/selectron?tab=readme-ov-file#contributing) more!)
 
## Use the CLI

```sh
# Install in a venv
uv add selectron
uv run selectron

# Or install globally
pipx install selectron
selectron
```

## Use the library

### Parse HTML

```python
from selectron.lib import parse
# ... get html from browser ...
res = parse(url, html)
print(json.dumps(res, indent=2))
```

If a parser is registered for the url, you'll receive something like this:

```json
[
  {
    "primary_url": "/_its_not_real_/status/1918760851957321857",
    "datetime": "2025-05-03T20:13:30.000Z",
    "id": "1918760851957321857",
    "author": "@_its_not_real_",
    "description": "\"They're made out of meat.\"\n\"Meat?\"\n\"Meat. Humans. They're made entirely out of meat.\"\n\"But that's impossible. What about all the tokens they generate? The text? The code?\"\n\"They do produce tokens, but the tokens aren't their essence. They're merely outputs. The humans themselves",
    "images": [{ "src": "https://pbs.twimg.com/profile_images/1307877522726682625/t5r3D_-n_x96.jpg" }, { "src": "https://pbs.twimg.com/profile_images/1800173618652979201/2cDLkS53_bigger.jpg" }]
  }
]
```

### Other functionality

The [selectron.chrome](https://github.com/SubstrateLabs/selectron/tree/main/src/selectron/chrome) and [selectron.ai](https://github.com/SubstrateLabs/selectron/tree/main/src/selectron/ai) modules are useful, but still baking, and subject to breaking changes – please pin your minor version. 

## Contributing

Generating parsers is easy by design:

1. Clone the repo
2. Run the CLI (`make dev`). Connect to Chrome.
3. In Chrome, open the page you want to parse. In the CLI, describe your selection (or use the AI-generated proposal).
4. Start AI selection (you can stop at any time to use the current highlighted selector).
5. Start AI parser generation. The parser will be saved to the appropriate location in `/src`. 
6. Review the parser's results and open a PR (please show what the parser produces).

### Setup

```sh
make install
make dev
# see Makefile for other commands
# see .env.EXAMPLE for config options
```
