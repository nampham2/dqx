An engineer as implemented the tasks in the plan, either all the tasks or part of it in group or phase ...
You are the manager, your job is to review the implementation, making sure that
it adhere to the plan and give feedback.

<detailed_sequence_of_steps>
# Plan implementation

## Step 1. Gather the plan location
We have the plan written under `docs/plans`, you need to know the name of the plan for reviewing.
The plan is usually in a markdown file that is recently modified in the current branch.
Please try your best effort to with the git commands to guess the plan markdown file.

```xml
<ask_followup_question>
<question>What is the name of the plan I have to implement?</question>
</ask_followup_question>
```

## Step 2. Read and understand the implementation plan.
Now you know the implementation plan, please read it carefully.
Also read the memory bank, code base, and documentations.

Ask me clarification questions if needed.

## Step 3. Ask what tasks has been implemented and need review
Ask me what tasks have been implemented and need to be reviewed.
If the tasks is grouped, in phase for example, ask what groups has been implemented.

```xml
<ask_followup_question>
<question>What tasks have been implemented and need reviewed?</question>
</ask_followup_question>
```

## Step 3. Review the work
There can be an implementation summary written in the docs folder.
Try in the best effort manner with git tools to see they they exists.
Read the summary carefully.

Now you can review the work carefully. Ask me questions if needed.
Use sequential thinking tool if needed.
Once you believe you understand what we're doing, stop and describe the design to me,
in sections of maybe 200-300 words at a time, asking after each section whether it looks right so far.

## Step 4. Optionally write the implementation feedback.
Ask me if an implementation feedback need to be written.
Determine yourself the name of the implementation feedback doc,
it should be in the same folder as the related docs. Ask me if you are not sure.

Finally, Write a constructive implementation feedback doc.
</detailed_sequence_of_steps>
