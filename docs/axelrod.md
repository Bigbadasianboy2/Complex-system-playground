# Axelrod Model

**Idea (in one breath):** Each agent has a small set of cultural features. Interaction follows the rules of homophily and social influence. Specifically, whether an interaction take place depends on agent's cultural similarity with its neighbor, and after an interaction agents become more similar. The dynamics is governed by a coarsening process of cultural domains (geometrically adjacent agents sharing the same culture configuration)

**What to look for**
- Small number of traits (q) → tends to consensus.
- Larger q → persistent fragmentation (many domains).
- Non-equilbrium phase transiton with respect to q. Order parameter: S_max(area of maxium domain)


**Result (example)**
[Axelrod domains](assets/images/axelrod_snapshots_new.pdf), [Axelrod phase transition](assets/images/Axelrod_Phase_Transition_F_vs_q.pdf)

**Try it**
See `models/Axelrod_lattice.snapshots.py` for domain coarsening, and see 'models/AXelrod_phase_transition.py' for the phase transitoon
