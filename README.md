Transfer learning added to the architecture of generalised AlphaZero. Adapted from code provided by Adrian Goldwaser. 
---

Methods: 

Train generalised AlphaZero (GAZ)

```
Usage: gaz_train.py <base_game> <train_to>
```
Evaluate GAZ
```
Usage: gaz_eval.py <base_game> <start> <finish>
```
Train transfer agent (single network transfer/apply to mimic networks)
```
Usage: transfer_train.py <base_game> <ckpt> <to_game> <mode> <train_to> - mode is 0pad, mean, map or clear
```
Evaluate transfer agent (any transfer method)
```
"Usage: transfer_eval.py <base_game> <start> <finish> <postfiz/path> <mode> - mode is simple or complex"
```
Generate expert buffers for multitask training method
```
"Usage: generate_buffer.py <game> <ckpt> <total_rounds>"
```
Multitask training method to generate mimic network for multi-network transfers
```
all settings in training script
```

Generating propnets
---
Propnets must be generated using the external package ggplib and it's dependency k273. These must be run on linux machines. Doc is hidden at this link. 

https://github.com/richemslie/ggplib/tree/master/doc


---
Original README: 

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
