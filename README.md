Alpha zero applied to General Game Playing
---

This repo contains a collection of GGP players and training scripts to use the techniques from AlphaZero for general game playing. It is part of my honours project at the University of New South Wales.

**Generalisations from AlphaZero:**
- No human hand-crafted network
- Multiplayer and single player games allowed
- Non-zero sum games allowed
- Simultaneous play between players allowed
- Non board games allowed

```
python3 b1train.py <game>  # to train a model
python3 b1vhuman.py <game> <role> # to play against most recent model
```
