# Complex-system-playground

This project Reproduces classic models in statistical physics & sociophysics, specifically:
- **Ising model (1D/2D)** — A Simple spin model. The corestone and the most classic example in complex systems and statistical physics. This project covers the reproduction of the order-disorder phase transition in 1D and 2D using Metropolis Hasting algorithms 
- **Axelrod model** — A spin-like Agent based model proposed by Robert Axelrod in 1997 to explain cultural dissemination. An non-equlibrium phase transition from a monocultural global consensus state to a multicultural fragmented state has been discovered by the statistical community. This project reproduces this phase transition and also visualizes the coarsening process of the model.  
- **NaSch traffic model** - A cellular automata that models traffic flow. This work produces the density–flow diagrams and the phase transition from the jam to free flow state. 

## Quickstart
```bash
git clone https://github.com/<Nianzheng Mao>/complex-systems-playground
cd complex-systems-playground
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
jupyter lab  # open examples/*.ipynb
