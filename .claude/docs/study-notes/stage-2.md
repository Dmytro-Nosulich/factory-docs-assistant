# Study explanation format

After each finished task, print this explanation in chat. Do not save explanations in this file.

Write in plain language and explain each new term the first time it appears. Core AI tasks (chunking, search, reranking, answers, evals) can run to about a page. Setup tasks stay short: skip any heading that has nothing useful to say.

Examples must use real documents and real outputs from this project. If something has not been run yet, say that it is an illustration. Never present an illustrative number as a result.

```markdown
## S2-XX Task name (YYYY-MM-DD)
**What it is and why it's needed:** the problem this step solves, what it builds on, and what the next step needs from it.
**How it works:** the core idea in plain language, with an analogy if one helps.
**Real example from our documents:** the actual input and output for one case.
**Why it matters:** what goes wrong without it, or when it is done badly.
**Settings and trade-offs:** each main setting, and what happens if it is raised or lowered.
**Watch out for:** common mistakes and how to spot them.
**What we decided and why:** the choices made and the alternatives we rejected.
**What we built:** what now exists, without code.
**Tech in brief:** tools and key settings, in a few lines.
**How to read the results:** what each metric means, what counts as good, and what to try if it is low.
**Results:** measured numbers only, each with the command that reproduces it.
```
