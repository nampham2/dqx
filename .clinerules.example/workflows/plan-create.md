# Implementation Plan Creation

Create a concrete implementation plan.

## Step 1: Determine Plan Version
Use git commands to identify the latest plan version in the current branch. Plans are typically recent markdown files. Choose whether to create a new version or revise the latest one.

<ask_followup_question>
<question>What version of the plan should I write?</question>
</ask_followup_question>

## Step 2: Write Comprehensive Plan
Write a detailed implementation plan in docs/plans/ using the version from Step 1.

Requirements:
- Assume the engineer has no context about the codebase
- Document all necessary information: files to modify, code samples, testing approach, and relevant documentation
- Structure tasks in groups of 3-5 for batch implementation
- Follow principles: DRY, YAGNI, TDD, frequent commits
- Final step must run pre-commit and pytest
- Only commit when all tests pass

The plan should guide a skilled developer who lacks domain knowledge through bite-sized, well-defined tasks.

## Step 3: Commit to git
### Step 3.1: Making sure the git-tree is clean
Check the current git branch:
  - If the current git tree is not clean, **ask me to clean up the git tree and return to the master branch**. Repeat checking until the git is on master branch and clean.
  - If on the master branch, create a new branch with a descriptive name. Consider whether to commit current changes before proceeding.

    <ask_followup_question>
    <question>Which git branch should I create?</question>
    </ask_followup_question>

### Step 3.2: Making the first commit with git

- Check out the branch with the branch name obtained in *Step 3.1*
- Check in in the plan written in *Step 1* and commit with a conventional commit format.
