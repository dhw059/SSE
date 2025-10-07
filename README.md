# Assessment and Application of Universal Machine Learning Interatomic Potentials in Solid-State Electrolyte Research

High-performance solid-state electrolytes (SSEs) are crucial for next-generation lithium batteries. However, conventional methods like density functional theory and empirical force fields face challenges in computational cost, scalability, and transferability across diverse systems. Machine learning interatomic potentials (MLIPs) offer a promising alternative by balancing accuracy and efficiency. Nevertheless, their performance and applicability for SSEs remain poorly defined, limiting reliable model selection. In this study, we benchmark 12 MLIPs─including GRACE, DPA, MatterSim, MACE, SevenNet, CHGNet, TensorNet, M3GNet, and ORB─across energies, forces, phonons, electrochemical stability, thermodynamic properties, elastic moduli, and Li+ diffusivity. GRACE-2L-OAM, MACE-MPA, MatterSim, DPA-3.1-3M, and SevenNet-MF-ompa show superior accuracy. Using MatterSim, we study Li3YCl6 and Li6PS5Cl, revealing that ∼40–50% S/Cl anion disorder enhances Li+ migration connectivity in Li6PS5Cl, while higher Li+ content in Li3Ycl6 expands conduction channels and reduces energy barriers. These insights highlight the power of MLIP-driven simulations for mechanistic understanding and rational design of high-conductivity SSEs.
## Model comparison and analysis

<img src="figs/3.png" width="1200">

<img src="figs/5.png" width="600">

Data and Code Availability
The computational workflows and analysis in this study leverage the MatCalc software package, a Python library for calculating materials properties from the potential energy surface (PES), including thermodynamic stability, phase diagrams, and mechanical properties. MatCalc was used to compute thermodynamic quantities and assess phase stability from MLIP-generated energy data.

If you use MatCalc, please cite it as follows:

Liu, R., Liu, E., Riebesell, J., Qi, J., Ong, S. P., & Ko, T. W. (2024). MatCalc: A Python library for calculating materials properties from the potential energy surface (PES) (Version 0.0.4). https://doi.org/10.5281/zenodo.xxxxxxx [replace with actual DOI if available]

Software available at: https://github.com/materialsvirtuallab/matcalc
