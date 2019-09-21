# Machine learning models for predicting RNA properties

<!-- <img src="app/static/logo_transparent.png" alt="drawing" width="200"/> -->


Options
-----------------------------------------------------------------------------------------------
The main scripts listed below each have options (with defaults). To see the options type

```
python [one_of_the_scripts_below.py] --help
```

Train and deploy an LSTM model for predicting the H-bond contact map for a sequence
-----------------------------------------------------------------------------------------------

First, prepare the experimental data for training:

```
python data/prepare_training_data.py
```

You may play with the encoder/decoder here:

```
python utils/sequence_encoder.py data/training/sequences.txt AGUCAGUC
```

which will print some statistics for you about the data set in addition to the example you supply.
-----------------------------------------------------------------------------------------------

=============================================
