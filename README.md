# Deep Distance Sensitivity Oracles 

This is the code implementation of the paper Deep Distance Sensitivity Oracles by Davin Jeong, Allison Gunby-Mann, Sarel Cohen, Maximilian Katzmann, Chau Pham, Arnav Bhakta, Tobias Friedrich, Sang Chin, originally published in Complex Networks 2023: https://arxiv.org/abs/2211.02681

## Setting up Environment

Experiments were run on Quadro RTX 8000 and an Intel(R) Xeon(R) Silver 4214R CPU, using cudatoolkit 10.1. Due to compatibility issues, we download each package manually; please refer to the commands in environment.txt.  

Additionally, please follow the following instructions to set up the "Graph Embedding" package for using node2vec (not affiliated with the authors of this paper). We assume that you begin at the root directory. 

`cd embeddings/GraphEmbedding`

`python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`

`sudo apt-get install gcc`

`sudo apt-get install g++`

`python3 -m pip install cython`

`python setup.py install`

## Reproducibility

Due to size constraints, all the datasets are not included in the supplementary material submission. Information on the used datasets are below, with the datesets included in the zip fold being bolded. The data refers to the largest connected component of each network, as we are interested in computing shortest paths between nodes, and a node's connected component can easily be stored and queried efficiently. 

| **Network Name **               | **Nodes** | **Density** | **Average Degree ** | **Dataset Size** | **MRE (Random Pivots)** | **MRE (Our Method)** | **Representation Factor** | **Link**                                                     |
|---------------------------------|-----------|-------------|---------------------|------------------|-------------------------|----------------------|---------------------------|--------------------------------------------------------------|
| chem-ENZYMES-g118               |        95 |    2.71E-02 |               2.547 |         9.03E+03 |                 215.96% |                0.27% |                    811.89 | https://networkrepository.com/ENZYMES-g118.php               |
| chem-ENZYMES-g296               |       125 |    1.82E-02 |               2.256 |         1.56E+04 |                 257.69% |                0.49% |                    530.22 | https://networkrepository.com/ENZYMES-g296.php               |
| infect-dublin                   |       410 |    3.30E-02 |              13.488 |         1.00E+05 |                 191.02% |                0.18% |                  1,091.55 | https://networkrepository.com/infect-dublin.php              |
| bio-celegans                    |       453 |    1.98E-02 |               8.940 |         1.00E+05 |                 176.28% |                0.04% |                  4,299.51 | https://networkrepository.com/bio-celegans.php               |
| **bn-mouse-kasthuri-graph-v4 ** |       987 |    3.16E-03 |               3.112 |         1.00E+05 |                 204.20% |                0.04% |                  5,105.11 | https://networkrepository.com/bn-mouse-kasthuri-graph-v4.php |
| can-1072                        |     1,072 |    1.18E-02 |              12.608 |         1.00E+05 |                 211.58% |                0.37% |                    573.40 | https://networkrepository.com/can-1054.php                   |
| scc_retweet                     |     1,150 |    9.98E-02 |             114.713 |         1.00E+05 |                 167.64% |                0.05% |                  3,287.11 | https://networkrepository.com/scc-retweet.php                |
| power-bcspwr09                  |     1,723 |    2.78E-03 |               4.779 |         1.00E+05 |                 237.46% |                0.16% |                  1,493.46 | https://networkrepository.com/power-bcspwr09.php             |
| inf-openflights                 |     2,905 |    3.71E-03 |              10.771 |         1.00E+05 |                 202.05% |                0.72% |                    280.63 | https://networkrepository.com/inf-openflights.php            |
| inf-power                       |     4,941 |    5.40E-04 |               2.669 |         1.00E+05 |                 228.67% |                0.23% |                  1,013.10 | https://networkrepository.com/inf-power.php                  |
| ca-Erdos992                     |     4,991 |    5.97E-04 |               2.977 |         1.00E+05 |                 206.26% |                0.33% |                    630.77 | https://networkrepository.com/ca-Erdos992.php                |
| **power-bcspwr10**              |     5,300 |    9.66E-04 |               5.121 |         1.00E+05 |                 232.83% |                0.32% |                    739.13 | https://networkrepository.com/power-bcspwr10.php             |
| bio-grid-yeast                  |     6,008 |    8.70E-03 |              52.245 |         1.00E+05 |                 173.80% |                6.25% |                     27.80 | https://networkrepository.com/bio-grid-yeast.php             |
| **soc-gplus**                   |    23,613 |    1.41E-04 |               3.319 |         1.00E+05 |                 200.20% |                0.31% |                    654.61 | https://networkrepository.com/soc-gplus.php                  |
| ia-email-EU                     |    32,430 |    1.03E-04 |               3.355 |         1.00E+05 |                 202.93% |                0.94% |                    216.11 | https://networkrepository.com/ia-email-EU.php                |
| ia-wiki-Talk                    |    92,117 |    8.50E-05 |               7.833 |         1.00E+05 |                 203.29% |                7.67% |                     26.50 | https://networkrepository.com/ia-wiki-Talk.php               |
| **tech-RL-caida**               |   190,914 |    3.33E-05 |               6.365 |         1.00E+05 |                 206.02% |                7.91% |                     26.04 | https://networkrepository.com/tech-RL-caida.php              |
|                                 |           |             |                     |                  |                         |                      |                           |                                                              |
|                                 |           |             |                     |                  |                         |                      |                           |                                                              |

1. Download the relevant datasets
2. Please comment out the lines for each network file before the list of edges. For an example, please look at graphs/network_repository/power-bcspwr10.mtx
3. Place relevant datasets in the list on line 17 of main.py and randomPivotTest.py
4. Run `python main.py`  and `python randomPivotTest.py`, which test our method and random pivots respectively

## Results

Please check log.txt for a summary
