# Redaction Notice

This is an anonymous review release. To protect unpublished implementation
know-how during double-blind review, the repository intentionally omits the
internal implementation of:

- Fairness-Guided Construction (FGC);
- Residual Reward Recovery (RR);
- fairness-preserving recovery/action-filtering logic; and
- internal rollout tie-breaking/search heuristics.

The public interfaces, neural backbone, data protocol, metrics, configuration,
and repository layout are included. The omitted components are represented by
explicit interfaces in `flog/decoder.py`; they fail loudly rather than silently
substituting a different algorithm.

The complete implementation can replace these protected interfaces in the full
public artifact without changing the surrounding repository structure.
