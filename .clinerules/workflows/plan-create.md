We have come up with a good implementation plan in mind, now we need to write the concrete plan.

<detailed_sequence_of_steps>
# Write out an implementation plan - Detailed sequence of steps.

## Step 1. Gather version information
Get the version information for the plan.
The plan is usually in a markdown file that is recently modified in the current branch.
Please try your best effort to with the git commands to guess the latest plan version.
You can either write a new plan version or revise the latest plan version.

<ask_followup_question>
<question>What version of the plan you want to write?</question>
</ask_followup_question>

## Step 2. Check out the git branch
Check out the current git branch, if it is the master branch then create a new branch with a suitable name.
Try to come up with the best git branch strategy, for examples use current branch, create a new branch,
commit the current changed files before moving on ...

<ask_followup_question>
<question>What git branch you want to create?</question>
</ask_followup_question>

## Step 2. Write a plan to docs/plans with version gathered in step 1.
Great. I need your help to write out a comprehensive implementation plan.

Assume that the engineer has zero context for our codebase and questionable taste. document everything they need to know. which files to touch for each task, code, testing, docs they might need to check. how to test it.give them the whole plan as bite-sized tasks. DRY. YAGNI. TDD. frequent commits.
Assume they are a skilled developer, but know almost nothing about our toolset or problem domain. assume they don't know good test design very well.

Plan requirement:
  - The last step of the plan is to run pre-commit, pytest and fix problems with them.
  - Only git commit the changes if all tests are passed and no problems with pre-commit. DO NOT commit in if some tests are expected to fail.

Please write out this plan, in full detail, into docs/plans/ with the version gathered in Step 1.
</detailed_sequence_of_steps>
