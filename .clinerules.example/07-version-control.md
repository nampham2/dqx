# Version Control - Git Standards

## Git Workflow

- Commit messages MUST follow conventional commit format. Read `https://www.conventionalcommits.org/en/v1.0.0/` with Context7 tool if needed.
- If the project isn't in a git repo, STOP and ask permission to initialize one.
- YOU MUST STOP and ask how to handle uncommitted changes or untracked files when starting work.  Suggest committing existing work first.
- When starting work without a clear branch for the current task, YOU MUST create a WIP branch.
- YOU MUST TRACK All non-trivial changes in git.
- YOU MUST commit frequently throughout the development process, even if your high-level tasks are not yet done. Commit your journal entries.
- NEVER SKIP, EVADE OR DISABLE A PRE-COMMIT HOOK. If there are problems with pre-commit hooks, fix and commit again.
- NEVER use `git add -A` unless you've just done a `git status` - Don't add random test files to the repo.
- Use `--no-pager` option with git to prevent git from waiting for my input. For example: `git --no-pager log --oneline -3`
