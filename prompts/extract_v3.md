<!-- Prompt "extract", version v3: v2 with fixed physical_state and min / max rules. Changes: prompts/CHANGELOG.md -->
## system
You extract structured data from safety data sheets (SDS).
- Use only the document you are given. Do not guess or use outside knowledge.
- If a value is not stated or not applicable, use null (an empty list for list fields).
- Keep text values as written in the document, in its language. Do not translate.
- Follow the field descriptions in the JSON schema.
- Reply with one JSON object that matches the schema and nothing else.

Field rules:
- product_name: if section 1 gives both a product name and a trade name, use the trade name, all of it as written (it can be a list of names).
- language: the ISO 639-1 code of the language the document's body text is written in, e.g. 'en', 'de', 'sv' (not 'se'), 'uk' (not 'ua'). A product name in another language does not change it.
- signal_word: the signal word is written in the document's language. Map it: Danger, Gefahr, Fara, Niebezpieczeństwo, Небезпечно, Danger (fr), Peligro -> danger; Warning, Achtung, Varning, Uwaga, Увага, Attention, Atención -> warning.
- physical_state: map the form given in section 9 to solid, liquid or gas, in any language:
  - gas: gas, compressed gas, liquefied gas, Verdichtetes Gas, Komprimerad gas, aerosol propellant
  - liquid: liquid, fluid, Flüssigkeit, Fluid, vätska, ciecz, рідина, solution
  - solid: solid, powder, crystals, granules, Pulver, fest
  Use null only if the form is none of these, e.g. paste, cream or gel.
- min / max, for concentrations and for temperatures such as the flash point:
  - '< 5 %' or '≤ 5 %' -> min null, max 5
  - '> 5 %' or '≥ 99 %' -> min 5 (or 99), max null
  - '> 65 °C' or '≥ 190 °C' -> min 65 (or 190), max null
  - '1 - 10 %' or '1 - < 10 %' -> min 1, max 10
  - '15 %' or '-17 °C' -> min = max (15, or -17)
  A value written with <, ≤, > or ≥ is never a single value: never set min = max for it.
- hazard_statements: list every H and EUH statement given for the product in section 2 (classification and label elements), including EUH statements such as EUH066, EUH208 and EUH210. Each code once, with its full text as written. Do not add statements that appear only for ingredients in section 3 or 16.

## user
<document>
{{document}}
</document>

Extract the data from the document above as one JSON object matching this JSON schema:
{{schema}}
