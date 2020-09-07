# EMMA
### Prerequisites
* Ubuntu 18.04
* Python 3.6
* CUDA 11.0

You can use pip to install Python depedencies.
```pip install -r requirements.txt ```
### Usage

* Step 1 training data 

First, you have to prepare the noisy audio files(*.wav), clean audio files(*.wav) and EMMA data(*.mat). and put the EMMA data under the same folder as clean audio files. In this step, we will transfer the audio file to spectrum and split the data in every 64 frames. The command line is displayed below:
```
python gen_pt.py --noisy_path < noisy_path (only needed in denoise mode )> \   
    --clean_path < clean_path > \       
    --out_path <out_path> \       
    --pwg_path <parallel_wavegan_path> \       
    --task < denoise / synthesis>
```
  In synthesis task, the model input only use EMMA data, hence, noisy_path is not needed. 
* Step 2 training mode
```
python main.py --mode <train / test> \
            --train_path <train_path>\
            --test_noisy < test_noisy >\
            --test_clean < test_clean > \
            --writer <log_path> \
            --model BLSTM_05 \
            --task < denoise / synthesis>
```
### Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
