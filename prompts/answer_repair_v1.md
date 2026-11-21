<!-- Prompt "answer_repair", version v1: one repair attempt for an answer that failed validation or the citation checks (S2-13). Sent after the answer conversation and the invalid reply. Changes: prompts/CHANGELOG.md -->
## user
Your answer did not pass validation:
{{errors}}

Fix these errors and reply with the complete corrected JSON object, nothing else. Change only what the errors require; keep every other value as it was. Use only the context. Every citation must name the `id` of a chunk in the context, and its quote must be copied character for character from that chunk's text. If no chunk supports a claim, remove the claim.
