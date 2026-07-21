---
name: prune-comments
description: Review code you just wrote or modified in this conversation and remove bad comments. Silently deletes AI-generated noise; flags human-written redundant comments for confirmation. Use after completing a coding task, or when the user says "prune comments", "clean up comments", or similar.
---

# Prune comments from recent changes

Scan the code you touched in this conversation and remove comments that don't earn their keep. This skill is about **your own output** — only operate on code you wrote or modified in the current session. Don't reach into unrelated files or unchanged code.

## What to remove (silently)

Delete these without asking:

1. **Top-of-file headers** that explain what the file does. Not this project's style — the filename and type declarations speak for themselves.
2. **Redundant comments** that restate the code beneath them. If a well-named function or variable already communicates the intent, the comment is noise.
3. **Conversation-reflective comments** that reference the current task, PR, discussion, or fix rather than the long-term nature of the code. Examples:
   - `// Added per user request`
   - `// Fix for the auth bug`
   - `// Handles the case from rdar://...`
   - `// This was refactored from the old approach`
4. **Multi-line comment blocks** or docstring novels on internal types where a one-liner (or nothing) would suffice.

## What to flag (not remove)

If you encounter a comment that appears **redundant but was clearly written by a human** (i.e., it existed before your changes, or its phrasing doesn't match your style), don't delete it. Instead, mention it to the user — they may know something you don't. Phrase it as: "This comment looks redundant to me — want me to remove it, or does it carry context I'm missing?"

## What to keep (never touch, maybe sometimes revise)

- **`// MARK:`** section dividers
- **`// TODO:` and `// FIXME:`** (these are fine in limited use)
- **Legal/license headers** (copyright notices, SPDX identifiers)
- **Tool directives** — `// swiftlint:disable`, `// sourcery:`, `// SAFETY:`, `// swiftformat:`, or any comment that controls tooling behavior. These are functional, not documentation.
- **Hidden constraints, subtle invariants, workarounds** — comments that explain *why* something non-obvious must be the way it is.
- **Algorithmic walk-throughs** for _genuinely complex__ processes. These should be casual and conversational in tone, like explaining to a colleague. First-person is fine. Numbered steps work well as waypoints through multi-phase logic. (See the sticky header layout code in Pegasus for the gold standard of this style.)

## Doc string rules

Doc strings (`///`) scale with access level:

- **`internal`** — optional. Only add one if the name alone is genuinely unclear.
- **`package`** — more common. A one-liner is usually enough.
- **`public`** — should generally exist. Keep them short and non-redundant.

In all cases, doc strings should be concise. Don't write multi-paragraph docstrings unless the task is specifically dedicated to documentation work.

**Length calibration:** A doc comment should be ONE line in many cases. Three or more lines often means you're over-explaining — tighten or delete. 

There can be a tendency to assume all situations are unique, special, and confusing. That's often not the case. If the doc comment restates information already obvious from the declaration (parameter names, return type), it adds nothing. A one-liner that says something the signature doesn't is better than a three-liner that paraphrases it. 

Bad (too verbose):
```swift
/// Returns an `AvailableControlCommand` for this skill, or `nil` if the skill
/// has opted out of slash-command surfaces via `isUserInvocable == false`.
public var availableControlCommand: AvailableControlCommand? {
```

Good (one line, adds the non-obvious part):
```swift
/// Returns an `AvailableControlCommand` for this skill, or `nil` if opted out.
public var availableControlCommand: AvailableControlCommand? {
```


## Procedure

1. Review the code you wrote or modified in this conversation. Use your memory of what you changed — don't run diffs or grep the whole repo.
2. For each comment that matches the "remove" criteria, delete it silently. Clean up any resulting awkward whitespace (double blank lines, trailing spaces).
3. For each comment that matches the "flag" criteria, note it for the user.
4. Don't touch anything in the "keep" category.
5. Briefly report what you pruned (e.g., "Removed 3 redundant comments and a file header from `FooBar.swift`."). If you flagged anything, ask about those.

## Tone calibration

When in doubt about whether a comment adds value, apply this test: **would removing it confuse a future reader who has never seen this conversation?** If no, remove it. If yes (or maybe), leave it.

## Top of file headings

The top of a new Swift file should always look like this: 

```
//
//  SupplementalToolCallParserRegistry.swift
//  IDEIntelligenceChat
//
//  Created by Iminabo Roberts on 3/8/26.
//
```

Abstracted away, that is: 

```
//
//  <File Name>
//  <Framework Name>
//
//  Created by Iminabo Roberts on M/D/YY.
//
```

Do not deviate from this unless explicitly instructed to.
