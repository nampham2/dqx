# Coding Rules Consolidation Plan v1

## Overview
Consolidate the content from `.clinerules/coding_rules.md` into the existing numbered rule files (01-06) and create new numbered files (02, 07-09) to properly organize all rules by category.

## Prerequisites
- Working knowledge of the DQX project structure
- Understanding of the Cline rules organization system
- Familiarity with Git for version control

## Implementation Plan

### Phase 1: Create New Rule Files

#### Task 1.1: Create 02-foundational-rules.md
**File to create**: `.clinerules/02-foundational-rules.md`

**Content to include from `coding_rules.md`**:
- General Rules section
- Foundational rules section
- Our relationship section

**Structure**:
```markdown
# Foundational Rules & Collaboration

## General Rules
[Content from "General Rules" section]

## Foundational Rules
[Content from "Foundational rules" section]

## Our Relationship
[Content from "Our relationship" section]
```

#### Task 1.2: Create 07-version-control.md
**File to create**: `.clinerules/07-version-control.md`

**Content to include from `coding_rules.md`**:
- Version Control section

**Structure**:
```markdown
# Version Control - Git Standards

## Git Workflow
[Content from "Version Control" section]
```

#### Task 1.3: Create 08-testing.md
**File to create**: `.clinerules/08-testing.md`

**Content to include from `coding_rules.md`**:
- Test Driven Development section
- Testing section
- Unit tests section (from project specific rules)

**Structure**:
```markdown
# Testing Standards - TDD & Quality

## Test Driven Development (TDD)
[Content from "Test Driven Development (TDD)" section]

## Testing Standards
[Content from "Testing" section]

## Unit Test Guidelines
[Content from "Unit tests" section in project specific rules]
```

### Phase 2: Update Existing Rule Files

#### Task 2.1: Update 01-project-context.md
**File to update**: `.clinerules/01-project-context.md`

**Sections to add**:
- Documentation section (from project specific rules)
- Virtual environment section (from project specific rules)
- Examples section (from project specific rules)

**Insert after "Project Structure" section**:
```markdown
## Documentation Rules
[Content from "Documentation" in project specific rules]

## Virtual Environment
[Content from "Virtual environment" in project specific rules]

## Examples
[Content from "Examples" in project specific rules]
```

#### Task 2.2: Update 03-development-methodology.md
**File to update**: `.clinerules/03-development-methodology.md`

**Sections to add**:
- Proactiveness section
- Designing software section
- Design patterns and algorithms section

**Insert after "Incremental Complexity Strategy" section**:
```markdown
## Proactiveness
[Content from "Proactiveness" section]

## Designing Software
[Content from "Designing software" section]

## Design Patterns and Algorithms
[Content from "Design patterns and algorithms" in project specific rules]
```

#### Task 2.3: Update 04-coding-standard.md
**File to update**: `.clinerules/04-coding-standard.md`

**Sections to add**:
- Writing code section
- Naming conventions section
- Code comments section
- Additional code style practices

**Insert after "Best Practices and Patterns" section**:
```markdown
## Writing Code Principles
[Content from "Writing code" section]

## Naming Conventions
[Content from "Naming" section]

## Code Comments
[Content from "Code Comments" section]

## Project-Specific Code Style
[Additional items from "Code Style and best practices" in project specific rules]
```

### Phase 3: Create Workflow Tools File

#### Task 3.1: Create 09-workflow-tools.md
**File to create**: `.clinerules/09-workflow-tools.md`

**Content to include from `coding_rules.md`**:
- Issue tracking section
- Learning and Memory Management section

**Structure**:
```markdown
# Workflow Tools - Productivity & Memory

## Issue Tracking
[Content from "Issue tracking" section]

## Learning and Memory Management
[Content from "Learning and Memory Management" section]
```

### Phase 4: Update Error Handling File

#### Task 4.1: Update 05-error-handling.md
**File to update**: `.clinerules/05-error-handling.md`

**Section to add**:
- Systematic Debugging Process section

**Insert after existing content**:
```markdown
## Systematic Debugging Process
[Content from "Systematic Debugging Process" section including all 4 phases]
```

### Phase 5: Cleanup and Verification

#### Task 5.1: Remove Original File
**Action**: Delete `.clinerules/coding_rules.md`

#### Task 5.2: Verify Content Distribution
**Check that all sections have been moved**:
- [ ] General Rules → 02-foundational-rules.md
- [ ] Foundational rules → 02-foundational-rules.md
- [ ] Our relationship → 02-foundational-rules.md
- [ ] Proactiveness → 03-development-methodology.md
- [ ] Designing software → 03-development-methodology.md
- [ ] Test Driven Development → 08-testing.md
- [ ] Writing code → 04-coding-standard.md
- [ ] Naming → 04-coding-standard.md
- [ ] Code Comments → 04-coding-standard.md
- [ ] Version Control → 07-version-control.md
- [ ] Testing → 08-testing.md
- [ ] Issue tracking → 09-workflow-tools.md
- [ ] Systematic Debugging Process → 05-error-handling.md
- [ ] Learning and Memory Management → 09-workflow-tools.md
- [ ] All project specific rules → Distributed to relevant files

#### Task 5.3: Run Pre-commit and Final Checks
**Commands to run**:
```bash
# Check git status
git status

# Add all changes
git add .clinerules/

# Run pre-commit hooks
bin/run-hooks.sh

# Commit if all checks pass
git commit -m "chore: reorganize cline rules into numbered structure

- Create new files: 02-foundational-rules, 07-version-control, 08-testing, 09-workflow-tools
- Update existing files with relevant content from coding_rules.md
- Remove original coding_rules.md after consolidation
- Maintain all original content in better organized structure"
```

## Testing and Verification

### Manual Verification Steps:
1. Search for any unique phrases from `coding_rules.md` to ensure nothing was missed
2. Review each numbered file to ensure logical organization
3. Check that no content was duplicated across files
4. Verify file naming follows the pattern: `{number}-{descriptive-name}.md`

### Success Criteria:
- All content from `coding_rules.md` is preserved in numbered files
- No duplicate content exists
- Files are logically organized by topic
- All numbered files follow consistent formatting
- Original `coding_rules.md` is removed
- Git commit is clean with no uncommitted changes

## Notes for Implementation:
- When copying content, preserve all formatting including bold, italics, and code blocks
- Maintain the original indentation and list structures
- Keep examples and specific commands intact
- Do not modify the content, only reorganize it
- If a section logically fits in multiple places, choose the most appropriate single location
