This directory contains data processing scripts and training/decoding configs for
performing speech recognition using wav2letter++ on popular datasets.

## Preparing data
Each dataset contains `prepare_data.py` which prepares the Dataset and Tokens file and `prepare_lm.py` which prepares Lexicon and Language Model data. Each file in the directory has instruction on how to run the python script.

> [...]/prepare_data.py [OPTIONS ...]

> [...]/prepare_lm.py [OPTIONS ...]

## Training/Decoding

The configs for training and decoding can be found under `configs` folder. Make sure to replace `[...]` with appropriate paths.

To run training
> [...]/wav2letter/build/Train train --flagsfile train.cfg

To run decoding
> [...]/wav2letter/build/Decode --flagsfile decode.cfg


*Replace [...] with appropriate paths*
