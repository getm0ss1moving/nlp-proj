# AGENTS.md

Project operating rules for Codex and other coding agents.

## Git Workflow

- For every project edit, create a separate git worktree and work on a `codex/...` branch.
- Before committing, run the most relevant verification for the change.
- For documentation-only changes, at minimum check `git diff --check` and inspect the staged diff.
- Commit only after verification passes.
- Push committed work to GitHub after the commit is created.

## Documentation Log

- Keep `docu.md` updated with user questions and assistant answers that are relevant to this project.
- Exclude unrelated personal, transient, or off-project conversation from the log.
- When compacting context, preserve project decisions, commands, files, experiment results, Git status, blockers, and next actions.
