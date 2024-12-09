# DC-POI+

My pytorch implementation was inpired by the paper paper below:

> Yifang Qin, Hongjun Wu, Wei Ju, Xiao Luo, Ming Zhang (2023). A Diffusion model for POI recommendation



I propose an accelerated dual-core diffusion model for the next POI recommen- dations (DC-POI+). The model encodes category data using Bidirectional Encoder Representations from Transformers (BERTs) for natural language processing before applying graph encoding modules. I leverage the diffusion process for both location sequences and category sequences by introducing a preconditioning factor. This preconditioning helps speed up the diffusion process and converges the targeted dataset faster. I evaluate the model using real- world check-in data from Foursquare platforms. Ablation studies and hyper-parameter analysis provide a comprehensive evaluation of the model’s performance, and potential future improve- ments are discussed in the end.

## Environment Requirement
The code has been tested running under Python 3.9.13. The required packages are as follows:

- pytorch == 1.11.0
- torchsde == 0.2.4
- torch_geometric (PyG) == 2.0.4
- pandas == 1.5.3


## Running Example
For example, to generate Foursquare `Tokyo` data for DC-POI+, first change the working directory into `~/.data` and run:

```
python process_data_tky-and-nyc.py
which will generate processed data files under the directory ~/processed/tky/.
```

To conduct experiment on Foursquare `Tokyo`, run:
```
cd ./code
python main.py --dataset tky --batch 1024 --patience 10 --dropout
```
For more execution arguments of Diff-POI, please refer to `~/code/main.py` or run
```
python main.py -h
```
