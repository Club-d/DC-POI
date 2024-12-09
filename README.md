# DC-POI+

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
