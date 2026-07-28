# langchain-anakin — verification status

## Live-verified, not assumed

- `anakin-sdk` is **live on PyPI right now** (confirmed via
  `curl https://pypi.org/pypi/anakin-sdk/json` → 200, version `0.1.0`,
  published 2026-04-28, `repository` field correctly points to
  `Anakin-Inc/anakin-py`). `pyproject.toml` depends on it normally
  (`anakin-sdk>=0.1.0,<0.2.0`) — no local path override needed or present.
- **A standalone package is the upstream-endorsed path, not a workaround.**
  `Anakin-Inc/langchain-community` is a real fork of
  `langchain-ai/langchain-community` and already has three closed PRs
  against the real upstream monorepo:
  [#552](https://github.com/langchain-ai/langchain-community/pull/552),
  [#553](https://github.com/langchain-ai/langchain-community/pull/553),
  [#554](https://github.com/langchain-ai/langchain-community/pull/554)
  (all opened 2026-02-23, #554 stayed open until 2026-04-11). A LangChain
  maintainer closed #554 with:

  > We no longer accept additional integrations in the `langchain`
  > monorepo. [...] I suggest to: Create your own repository to distribute
  > LangChain integrations, Publish the package to PyPI.

  So this package's approach — standalone repo, publish to PyPI as
  `langchain-anakin` — is exactly what upstream told the team to do, not a
  parallel/duplicate effort. The `feat/anakin-integration` branch still
  exists on the fork with the original monorepo-shaped code
  (`libs/community/langchain_community/{document_loaders,tools,utilities}/anakin*`)
  if anyone wants to reference the original implementation, but it should
  not be resubmitted upstream in that form.
- All source files (`langchain_anakin/*.py`, `tests/test_tools.py`) pass
  `python3 -m py_compile` — syntactically valid.
- Every field/method referenced against `anakin-py`'s real source, not
  assumed: `Anakin.__init__(api_key=, *, base_url=, ...)`, `client.scrape()`,
  `client.search()`, `client.agentic_search()` signatures, and the
  `Document`/`SearchResult`/`SearchResultItem`/`AgenticSearchResult`/
  `AgenticSearchData` Pydantic model fields all read directly from
  `anakin-py/src/anakin/client.py` and `models.py` — not the public docs
  (which, per the Pipedream/n8n work in this same session, were found to
  omit real parameters like `forceFresh` and agentic search's `useBrowser`/
  `schema`). Both are included here.
- Tool structure follows `langchain-exa`'s real, current, in-tree pattern
  (pulled from `langchain-ai/langchain`'s actual repo via `gh api`, not
  memory) — `BaseTool` subclass, `model_validator`-based client
  construction, `_run` with typed args mirroring the underlying client
  call.

## What's NOT verified

`pip install` hung in this environment with near-zero CPU usage after
~10 minutes (checked via `ps` — the process was idle, not computing),
consistent with restricted network access in this sandbox rather than a
real problem with the dependencies. **Never got a live `pytest` run.**
Before trusting this package:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

The five tests in `tests/test_tools.py` mock the Anakin client directly
(no live API calls, no network needed beyond the `pip install` step) — they
should pass immediately once the environment can actually install
`langchain-core` and `anakin-sdk` (both are normal PyPI installs now).

## Steps (needs the account owner)

1. Run the install + test commands above in an environment with normal
   network access.
2. Push as `Anakin-Inc/anakin-langchain`, publish to PyPI as
   `langchain-anakin`.
3. Optional, once live: submit to LangChain's own docs
   (`docs.langchain.com`) as a community/partner integration listing —
   not investigated in this pass, same-shaped question as the Google ADK /
   Agno submissions elsewhere in this session (a docs PR, likely to
   `langchain-ai/langchain` or `langchain-ai/docs`).

## Not done

Never installed or run — verified by static reading against real source
only. Treat as reviewed-but-unexecuted, same status as the Pipedream
components before this note was written for them.
