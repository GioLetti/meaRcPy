# MeARcPy: Multielectrode Array Reservoir computing in Python

MeaRcPy is a novel software that analyzes electrophysiological multielectrode array recordings (MeA) of in-vitro neuronal networks. The model is based on the Reservoir Computer Network (Rc) architecture, which uncovers connectivity patterns among different sampled regions in the network and captures dynamic interactions between them. MeARcPy can be used to predict the network’s connectivity map and demonstrates the ability to forecast the spatio-temporal response of a given network to specific inputs.\
Moreover, MeARcPy contains a module for simulating neuronal network recordings based on numerical simulations conducted using the NEST simulator [1].\
The Rc model has been developed on top of the ReservoirPy library [2].

A complete description of the meaRcPy model can be found in the paper [3]:
**Decoding neuronal networks: A Reservoir Computing approach for predicting connectivity and functionality**

In the **Simulation module** code to perform NEST simulation is provided. The**analysis module** contains the code of the MeaRcPy model and to run the analysis over the electrophysiological and simulated data.

### Ref:
\
[1] M.-O. Gewaltig and M. Diesmann, “Nest (neural simulation tool),” Scholarpedia, vol. 2, no. 4, p. 1430, 2007.\
[2] N. Trouvain, L. Pedrelli, T. T. Dinh, and X. Hinaut, “ReservoirPy: An efficient and user-friendly library to design
    echo state networks,” in Artificial Neural Networks and Machine Learning – ICANN 2020, pp. 494–505, Springer International Publishing, 2020.\
[3] Auslender, I., Letti, G., Heydari, Y., Zaccaria, C., & Pavesi, L. (2024). Decoding neuronal networks: A Reservoir Computing approach for predicting connectivity and functionality. Neural Networks, 107058.
