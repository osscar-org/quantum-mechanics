# **Interactive Notebooks: Quantum Mechanics and Computational Materials Science**
[![Voila test](https://github.com/osscar-org/quantum-mechanics/actions/workflows/voila-test.yml/badge.svg)](https://github.com/osscar-org/quantum-mechanics/actions/workflows/voila-test.yml)

**Introduction**: This is a repository for tutorials in quantum mechanics and
computational materials science. Jupyter notebooks are converted into
interactive web applications. Professors can use them to demonstrate knowledge
in the classroom. Students can also use them for self-learning.

![example](./images/examples.png)

* Try on `Materials Cloud` (fast, a few seconds to load) [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io)

* Try on `BinderHub` (can take > 1 minute to load) [![badge](https://img.shields.io/badge/OSSCAR-binder-E66581.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Findex.ipynb)

## Local installation

One can clone this repository and install all the required packages locally.
To do so (ideally in a fresh python virtual environment, to reduce risks of version clashes of some of the dependencies), run the following commands:

```bash
git clone https://github.com/osscar-org/quantum-mechanics.git
cd quantum-mechanics
pip install -r requirements.txt
```

Then, to view the notebooks in the form of a web application, you can type
the following command in the terminal:

```bash
voila --template=osscar --VoilaConfiguration.enable_nbextensions=True notebook/
```

This will start the voila server and then open your default browser, where you can use the web application.

## Content

### Section 1: Quantum Mechanics

Here are the tutorials to demonstrate numerical solution for 1D Schrödinger
equation.

| Name       | Description           | Notebook links  | Binder | Materials Cloud | 
| ------------- |:-------------:| -----:| -----:| -----:|
| One Quantum Well | The solution for 1 quantum well | [One Quantum Well](./notebook/quantum-mechanics/1quantumwell.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fquantum-mechanics%2F1quantumwell.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/quantum-mechanics/1quantumwell.ipynb) |
| Two Quantum Wells | The solution for 2 quantum wells | [Two Quantum Wells](./notebook/quantum-mechanics/2quantumwells.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fquantum-mechanics%2F2quantumwells.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/quantum-mechanics/2quantumwells.ipynb) |
| Asymmetric Well | Avoided crossing in an asymmetric well | [Asymmetric Well](./notebook/quantum-mechanics/asymmetricwell.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fquantum-mechanics%2Fasymmetricwell.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/quantum-mechanics/asymmetricwell.ipynb) | 
| Shooting method | Shooting method with Numerov algorithm | [Shooting method](./notebook/quantum-mechanics/shooting_method.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fquantum-mechanics%2Fshooting_method.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/quantum-mechanics/shooting_method.ipynb) |
| SOFT | Split operator Fourier transform method | [SOFT](./notebook/quantum-mechanics/soft.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fquantum-mechanics%2Fsoft.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/quantum-mechanics/soft.ipynb) |
| MSOFT | Multiple Split operator Fourier transform method | [SOFT](./notebook/quantum-mechanics/msoft.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fquantum-mechanics%2Fmsoft.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/quantum-mechanics/msoft.ipynb) |

### Section 2: Band Theory of Crystals

Here are the tutorials to demonstrate the band theory of crystal systems.

| Name       | Description           | Notebook links  | Binder | Materials Cloud | 
| ------------- |:-------------:| -----:| -----:| -----:|
| FFT and Planewaves | Fourier Transforms and Plane-Wave Expansions | [FFT and Planewaves](./notebook/band-theory/FFT_and_planewaves.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fband-theory%2FFFT_and_planewaves.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/band-theory/FFT_and_planewaves.ipynb) |
| Free Electron | Free-electron Bands in a Periodic Lattice | [Free Electron](./notebook/band-theory/free_electron.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fband-theory%2Ffree_electron.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/band-theory/free_electron.ipynb) |
| Density of States | Density of States (DOS) | [Density of States](./notebook/band-theory/density_of_states.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fband-theory%2Fdensity_of_states.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/band-theory/density_of_states.ipynb) |
| Pseudopotentials | Norm-conserving pseudopotentials | [Pseudopotentials](./notebook/band-theory/pseudopotential.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fband-theory%2Fpseudopotential.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/band-theory/pseudopotential.ipynb) |


### Section 3: Molecule and Lattice Vibration
| Name       | Description           | Notebook links  | Binder | Materials Cloud | 
| ------------- |:-------------:| -----:| -----:| -----:|
| Phonon 1D | lattice vibration for one dimensional system | [Phonons 1D](./notebook/lattice-vibration/Phonon_1D.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Flattice-vibration%2Phonon_1D.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/lattice-vibration/Phonon_1D.ipynb) |
| Phonon 2D | lattice vibration for two dimensional systems | [Phonon 2D](./notebook/lattice-vibration/Phonon_1D.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Flattice-vibration%2Phonon_2D.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/lattice-vibration/Phonon_2D.ipynb) |
| Molecular Vibrations | introduce to molecular vibrations | [Molecular Vibration](./notebook/lattice-vibration/Phonon_1D.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Flattice-vibration%2Molecule_Vibration.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/lattice-vibration/Molecule_Vibration.ipynb) |

### Section 4: Molecular Dynamics

| Name       | Description           | Notebook links  | Binder | Materials Cloud | 
| ------------- |:-------------:| -----:| -----:| -----:|
| Verlet Integration | Verlet integration | [Verlet Integration](./notebook/molecular-dynamics/verlet_integration.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fmolecular-dynamics%2Fverlet_integration.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/molecular-dynamics/verlet_integration.ipynb) |

### Section 5: Statistical Mechanics

| Name       | Description           | Notebook links  | Binder | Materials Cloud | 
| ------------- |:-------------:| -----:| -----:| -----:|
| Metropolis Monte Carlo | Metropolis-Hastings Monte Carlo | [Metropolis Monte Carlo](./notebook/statistical-mechanics/monte_carlo_parabolic.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fstatistical-mechanics%2Fmonte_carlo_parabolic.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/statistical-mechanics/monte_carlo_parabolic.ipynb) |
| Monte Carlo Integration | Monte Carlo Integration | [Monte Carlo Integration](./notebook/statistical-mechanics/monte_carlo_pi.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fstatistical-mechanics%2Fmonte_carlo_pi.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/statistical-mechanics/monte_carlo_pi.ipynb) |
| Ising Model | Ising Model | [Ising Model](./notebook/statistical-mechanics/ising_model.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/osscar-org/quantum-mechanics/master?urlpath=%2Fvoila%2Frender%2Fnotebook%2Fstatistical-mechanics%2Fising_model.ipynb) | [![Materials Cloud Tool osscar-qmcourse](https://raw.githubusercontent.com/materialscloud-org/mcloud-badge/main/badges/img/mcloud_badge_tools.svg)](https://osscar-quantum-mechanics.materialscloud.io/voila/render/statistical-mechanics/ising_model.ipynb) |

## How to contribute

If you would like to contribute a new notebook to OSSCAR, see the [guide to contributing](https://www.osscar.org/code/contributing.html) on our website, where you can also find an example [notebook template](https://www.osscar.org/code/sample_notebook.html). 

## How to cite

When using the content of this repository, please cite the following article:

 D. Du, T. Baird, S. Bonella and G. Pizzi, OSSCAR, an open platform for collaborative development of computational tools for education in science, *Computer Physics Communications*, **282**, 108546 (2023).
https://doi.org/10.1016/j.cpc.2022.108546
## Acknowledgements

We acknowledge support from the EPFL Open Science Fund via the [OSSCAR](http://www.osscar.org) project.

<img src='https://www.osscar.org/_images/logos.png' width='700'>
