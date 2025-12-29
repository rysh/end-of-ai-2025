# Dissipative Intelligence

**From the "Logic Trap" to Nonlinear Causal Dynamics**

This repository contains the simulation code and LaTeX source for the paper arguing that current AI architectures (linear stack approximations) are fundamentally incapable of handling causal phase transitions, and proposing a dissipative intelligence model based on the Landau-Stuart equation.

## Paper Abstract

Current LLMs face a sustainability crisis due to the "Logic Trap" phenomenon. This paper provides mathematical proof for why linear accumulation fails at paradigm shifts, and proposes a shift from "Static Accumulation" to "Dissipative Intelligence."

## Files

| File | Description |
|------|-------------|
| `paper.tex` | LaTeX source for the paper |
| `paper.pdf` | Compiled PDF |
| `causal_dynamics_comparison.py` | Simulation comparing linear vs nonlinear models |
| `causal_dynamics_comparison.png` | Generated figure |

## Running the Simulation

```bash
pip install numpy matplotlib scikit-learn
python causal_dynamics_comparison.py
```

This generates a comparison plot showing:
- **Ground Truth**: Noisy reality following Landau-Stuart dynamics with Hopf bifurcation
- **Model A (Red)**: Linear stack approximation that overfits and diverges
- **Model B (Blue)**: Landau-Stuart dynamics that captures structural change

## Building the Paper

```bash
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
```

## Author

Ryuhei Ishibashi - Elanare Institute

## License

See [LICENSE](LICENSE) for details.
