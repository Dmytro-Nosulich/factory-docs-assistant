<!-- Prompt "repair", version v1: one repair attempt for a reply that failed validation (S1-15). Sent after the extraction conversation and the invalid reply. Changes: prompts/CHANGELOG.md -->
## user
Your reply did not pass validation against the schema:
{{errors}}

Fix these errors and reply with the complete corrected JSON object, nothing else. Change only what the errors require; keep every other value as it was. Use only the document.
