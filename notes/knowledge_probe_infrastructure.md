# Knowledge Probe Infrastructure — Research Program Notes

*Captured April 2026, from conversation during WWI corpus expansion.*

---

## The Core Idea

The redaction/recovery mechanism we built for NER recovery is not WWI-specific and not history-specific. It is a **domain-agnostic hallucination probe**: given a passage with named entities masked, can a language model recover what was removed? The answer reveals what the model has genuinely internalized versus confabulated.

The current instantiation (WWI Wikipedia → DistilBERT MLM) is a proof of concept. The architecture — corpus ingestion, redaction pipeline, constrained candidate eval — generalises cleanly to any domain with structured text and named entities.

---

## The Broader Claim

**Hallucination is domain-shaped.** Models fail differently in mathematics than in history than in philosophy. The failure modes are not random noise — they reflect the structure of training data, the density of co-occurrence, and the specificity of entity-concept bindings in each domain. A probe infrastructure that can be instantiated across domains and compared is a tool for *mapping* hallucination rather than merely detecting it.

---

## Domain Instantiations

### History
- **Current corpus:** WWI Wikipedia (3,998 articles, 81 MB). Expanding to Revolutionary War → WWI arc.
- **Primary source extension:** Avalon Project (treaties, war declarations), FRUS (State Dept cables), Founders Online (founding correspondence). Primary sources contain the ghost terms Wikipedia summarises away — minor figures, obscure places, single-mention entities. This is where the probe gets hard in a meaningful way.

### Mathematics
- **Sources:** MacTutor History of Mathematics, OEIS, arXiv math preprints.
- **Entity type:** Theorems, mathematicians, dates/venues of proof, journal attributions.
- **Interesting failure mode:** Attribution blending — the model knows a theorem exists but assigns it to the wrong person or era. Constrained eval against a per-mathematician candidate set would surface this directly.

### Philosophy
- **Sources:** Stanford Encyclopedia of Philosophy (SEP) — already structured, dense named entities, precise conceptual lineage.
- **Entity type:** Philosophers, works, arguments, dates of publication.
- **Interesting failure mode:** Did the model learn that Kripke's rigid designator argument appeared in *Naming and Necessity* (1980), or did it hallucinate a journal article? SEP's citation structure makes ground truth unambiguous.

### Science
- **Sources:** ORCID, Google Scholar, patent databases, arXiv.
- **Entity type:** Researchers, institutions, paper titles, discovery dates.
- **Interesting failure mode:** The long tail of real researchers the model may have seen once in a citation and half-remembered. High OOV rate expected — methodologically interesting.

---

## Why the OOV Rate Matters

Current test set OOV rate: **51.9%** (entities in test not seen in training candidates). This is not a failure of the model — it is a signal about the *knowledge boundary*. Primary sources and non-Wikipedia corpora would push OOV higher and make the constrained eval work harder in a meaningful way. The probe becomes most informative exactly where the model's coverage gets thin.

---

## Corpus Scope Discipline

Every domain has scope creep risk — the boundary always has something interesting just past it. The working principle: **draw the boundary where there is a coherent argument, not just where the interesting material runs out.**

For the history instantiation: Revolutionary War → WWI is the right arc. It traces American foreign policy from the first assertion of sovereignty to the war that ended American isolationism. That is a complete argument. Bronze Age Sea Peoples is a different paper.

---

## What This Could Be

Not a paper. A **research program** — potentially a shared evaluation infrastructure for hallucination probing across domains, with the current NER recovery work as the founding instantiation and proof of concept.

Possible name: *Engrammatic* (already in use for the current project) or something that gestures at the knowledge-trace / ghost-term framing.
