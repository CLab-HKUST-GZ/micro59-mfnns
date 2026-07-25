# Decision Log

## D01: distinguish figure reproduction from source-experiment rerun

The reviewer requirement is satisfied only when the repository can regenerate
and validate the submitted figures. It does not require silently claiming that
historical absolute-path YAMLs are portable. The documentation now separates:

1. deterministic figure reproduction from versioned evidence; and
2. optional, resource-intensive source reruns with explicit inputs.

## D02: preserve historical provenance YAMLs

Figure 18 and Figure 23 hashes are evidence. Rewriting those archived files
would destroy the provenance they are intended to preserve. Figure 18 instead
creates runtime copies; Figure 23 explicitly treats its YAMLs as read-only
historical records.

## D03: align the builder with hundreds of portable YAMLs

The shortest consistent policy is one normalized index per dataset in the flat
path already used by all portable Figure 14 and 21 configurations. Changing
372 YAMLs to a reintroduced variant directory would alter the archived
experiment contract and require duplicating normalized/raw query and
ground-truth semantics. The builder and cache README were therefore restored
to the flat normalized policy.

## D04: do not consume GPU resources

The user explicitly excluded GPU checks. The unified path regenerates Figure
18 from frozen data and only statically validates optional BANG producer
scripts. No GPU availability was inferred and no GPU command was launched.

## D05: preserve the original dirty workspace

The main workspace contained substantial pre-existing user work. Remote
inspection, changes, tests, and the commit were isolated in a clean temporary
worktree tracking `micro59/main`; no reset, checkout, or cleanup touched the
user's working files.
