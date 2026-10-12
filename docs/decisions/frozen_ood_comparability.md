# Retain the frozen OOD inputs for paper comparability

User selection, 2026-10-12: keep the existing six-source, 6,395-row OOD inputs
and reported scores unchanged so comparisons retain the paper-inspired input
contract. Defer assistant-prose restoration and corrected-input rescoring until
the user revisits this decision. Do not promote restoration artifacts into
benchmark defaults. Author contact is also deferred; this decision does not
authorize contacting anyone.

The frozen teacher input SHA256 is
`fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`;
the student input SHA256 is
`ee2baa18a5d36cda8f9d66df642b6f3c106f92ae8ce2813acca36fa9f0e4acd2`.
Preserve the [input audit](../findings/ood_input_reasoning.md) and
[prose/conversion findings](../findings/ood_prose_restoration.md) as limitations
on interpretation. This retains our frozen representation; it does not establish
the exact bytes used in every reported paper run.
