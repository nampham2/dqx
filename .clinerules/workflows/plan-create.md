# Implementation Plan Creation

Create a concrete implementation plan.

<detailed_sequence_of_steps>

## Step 1: Determine Plan Version
Use git commands to identify the latest plan version in the current branch. Plans are typically recent markdown files. Choose whether to create a new version or revise the latest one.

<ask_followup_question>
<question>What version of the plan should I write?</question>
</ask_followup_question>

## Step 2: Manage Git Branch
Check the current git branch. If on the master branch, create a new branch with a descriptive name. Consider whether to commit current changes before proceeding.

<ask_followup_question>
<question>Which git branch should I create?</question>
</ask_followup_question>

YOU MUST create a git branch based on the above gathered information.

## Step 3: Write Comprehensive Plan
Write a detailed implementation plan in docs/plans/ using the version from Step 1.

Requirements:
- Assume the engineer has no context about the codebase
- Document all necessary information: files to modify, code samples, testing approach, and relevant documentation
- Structure tasks in groups of 3-5 for batch implementation
- Follow principles: DRY, YAGNI, TDD, frequent commits
- Final step must run pre-commit and pytest
- Only commit when all tests pass

The plan should guide a skilled developer who lacks domain knowledge through bite-sized, well-defined tasks.

</detailed_sequence_of_steps>
