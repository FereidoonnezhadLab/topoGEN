# TopoGEN
TopoGEN is a framework that integrates three-dimensional image-informed fiber network generation with non-linear finite element analysis to support the mechanistic investigation of structure-function relationships in soft matter. 

![Abstract](abstract.svg)

## Requirements
- **Python**: pre-processing and data analysis
- **Fortran**: bilinear USDFLD fiber behavior
- **Abaqus Standard**: model solver

## Overview
The central component is the [`src`](./src) folder that implements the logic to generate the topologies and build the Abaqus input files for micromechanical modeling. Everything is embedded into one single main.py file. The auxiliary files in the source folder are intended to perform the following steps:

- **STEP 1**: GENERATION OF A PERIODIC VORONOI IN THE 3D SPACE ([network generation](src/create_periodic_network.py))

- **STEP 2**: OPTIMIZATION OF THE STRUCTURE AGAINST THE PARAMETERS  [network optimization](src/optimize_periodic_network.py)
- **STEP 3**: NETWORK REFINEMENT (REMOVES DANGLING ENDS AND UNCONNECTED EDGES)
- **STEP 4**: GENERATION OF THE MICROMECHANICAL TESTS TO BE RUN IN ABAQUS [Abaqus input files](src/write_abaqus_input_file.py)

The user can select the topological input or the range that they want to test and then the pipeline proceeds with the posprocessing of the simulation results. The folder [`analysis`](./analysis) contains functions to:
- quantify the impact of each microstructural parameter on the bulk mechanical properties [`microstructural effects`](./analysis/microstructural_effects/)
- study the nonaffine behavior of the internal nodes [`nonaffinity check`](./analysis/nonaffinity_check/)

## Contents
Typical layout:
- generator_file — main generator program (executable/script)
- config/ or config.yaml — configuration for generation runs
- data/ — input datasets or test inputs
- scripts/ — helper scripts used to prepare inputs or run batches
- analysis/ — extra analysis and plotting code
- tests/ — unit/integration tests
- requirements.txt or environment.yml — dependency list
- README.md — this file

(Adjust names to match the actual files in this folder.)

## Quick start
1. Install dependencies:
    - Python example:
      ```bash
      pip install -r requirements.txt
      ```
    - or use your preferred package manager / environment.

2. Run the generator:
    - If `generator_file` is a script:
      ```bash
      python main.py --config config/config.yaml --out results/
      ```
    - If it's executable:
    tbd

3. Examine outputs in the specified output directory (`results/` above).

## Configuration and inputs
- Use the config file (e.g. `config.yaml`) to set generation parameters: input sources, seeds, output formats, and algorithm options.
- Input data (if required) goes into `data/`; the generator reads these paths defined in the config.

## Contributing
Thank you for using TopoGEN! For any inquiries, additional help, customization, or any other problems/concerns/suggestions, please reach out to us via email. The author of this codes is Sara Cardona (s.cardona@tudelft.nl).

## Copyright
Copyright (c) Sara Cardona, PhD Researcher, ME, TU Delft (2025)

To Cite this work please cite one of these articles:
https://doi.org/10.1016/j.jmps.2025.106257
