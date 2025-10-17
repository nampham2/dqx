# Development Methodology - KISS/YAGNI

## 3D Methodology (Design-Driven Development)

Key principle: Think before you build, build with intention, ship with confidence.

### Quick 3D Reminders

- **Run `uv run pytest tests/ -v` at end of every phase** - 100% pass required
- **Update implementation progress checkboxes** as you complete each phase
- **Follow Example Creation Guidelines** for comprehensive examples
- **Apply Unit Testing Guidelines** for thorough test coverage

## KISS/YAGNI Design Principles

**KISS (Keep It Simple, Stupid)** & **YAGNI (You Aren't Gonna Need It)**: Balance engineering rigor with practical simplicity.

### AI Decision-Making Guidelines

🎯 **START SIMPLE, EVOLVE THOUGHTFULLY**

For design decisions, AI coders should:
1. **Default to simplest solution** that meets current requirements
2. **Document complexity trade-offs** when proposing alternatives
3. **Present options** when multiple approaches have merit
4. **Justify complexity** only when immediate needs require it

🤖 **AI CAN DECIDE** (choose simplest):
- Data structure choice (dict vs class vs dataclass)
- Function organization (single file vs module split)
- Error handling level (basic vs comprehensive)
- Documentation depth (minimal vs extensive)

👤 **PRESENT TO HUMAN** (let them choose):
- Architecture patterns (monolith vs microservices)
- Framework choices (custom vs third-party)
- Performance optimizations (simple vs complex)
- Extensibility mechanisms (hardcoded vs configurable)

⚖️ **COMPLEXITY JUSTIFICATION TEMPLATE**:
"Proposing [complex solution] over [simple solution] because:
- Current requirement: [specific need]
- Simple approach limitation: [concrete issue]
- Complexity benefit: [measurable advantage]
- Alternative: [let human decide vs simpler approach]"

### Common Over-Engineering Patterns to Avoid

❌ **AVOID** (unless specifically needed):
- Abstract base classes for single implementations
- Configuration systems for hardcoded values
- Generic solutions for specific problems
- Premature performance optimizations
- Complex inheritance hierarchies
- Over-flexible APIs with many parameters
- Caching systems without proven performance needs
- Event systems for simple function calls

✅ **PREFER** (start here):
- Concrete implementations that work
- Hardcoded values that can be extracted later
- Specific solutions for specific problems
- Simple, readable code first
- Composition over inheritance
- Simple function signatures
- Direct computation until performance matters
- Direct function calls for simple interactions

### Incremental Complexity Strategy

📈 **EVOLUTION PATH** (add complexity only when needed):

Phase 1: Hardcoded → Phase 2: Configurable → Phase 3: Extensible

Example:
- Phase 1: `return "Hello, World!"`
- Phase 2: `return f"Hello, {name}!"`
- Phase 3: `return formatter.format(greeting_template, name)`

🔄 **WHEN TO EVOLVE**:
- Phase 1→2: When second use case appears
- Phase 2→3: When third different pattern emerges
- Never evolve: If usage remains stable

## Proactiveness

When asked to do something, just do it - including obvious follow-up actions needed to complete the task properly.
  Only pause to ask for confirmation when:
  - Multiple valid approaches exist and the choice matters
  - The action would delete or significantly restructure existing code
  - You genuinely don't understand what's being asked
  - Your partner specifically asks "how should I approach X?" (answer the question, don't jump to
  implementation)

## Designing Software

- YAGNI. The best code is no code. Don't add features we don't need right now.
- When it doesn't conflict with YAGNI, architect for extensibility and flexibility.

## Design Patterns and Algorithms

- Employ design patterns when planning and act.
  Provide different solutions with different trade off and ask users to make decision on which solution to choose.
- Thinks about algorithm complexity and communicate clearly to users.
- Prefer simple, elegant solution over more complex ones.
- Always analyze code for potential duplicates and give proposal to remove them.
