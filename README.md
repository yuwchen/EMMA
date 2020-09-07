# EMMA
### Prerequisites
* Ubuntu 18.04
* Python 3.6
* CUDA 11.0

You can use pip to install Python depedencies.
```
pip install -r requirements.txt 
```
## Usage

### Step 1 prepare training data 

Transform waveform to spectrogram by STFT, and divide spectrogram by frame size 64. 

Setup your dataset to be the following structure.

```
/path/to/clean/
├── utt_1.wav
├── utt_1.mat
│   ...
│   ...
├── utt_N.wav
└── utt_N.mat
```
where *.wav is the clean audio file and *.mat is the corresponding EMMA data


#### speech enhancement
For speech enhancement, you need to prepare noisy dataset.
The noisy speech should have the same name as its corresponding clean speech. 
```
/path/to/noisy/data/
├── SNR1
│   ├── noise_type1
│.  │.  ├──utt_1.wav
├── SNR2
│   ├── noise_type2
│.  │.  ├──utt_1.wav
│   ...
└── SNR_N
    ├── noise_typeN
    │.  ├──utt_1.wav    
    
```
  
Then, generate the training data for speech enhancement.
```
python gen_pt.py --noisy_path <noisy_path> \   
    --clean_path <clean_path> \       
    --out_path <out_path> \       
    --task <denoise>
```

#### speech synthesis

Generate the training data for speech synthesis.

```
python gen_pt.py --clean_path <clean_path > \       
    --out_path <out_path> \       
    --pwg_path <parallel_wavegan_path> \       
    --task <synthesis>
```


  
### Step 2 use pre-trained model or train your own model

In this step, we used the data generated before to train the model. same as the first step, the model input only use EMMA data, <test_noisy> is not needed. we offer several example model structure. Go check ```/main/run.sh``` for further information
```
python main.py --mode <train / test> \
            --train_path <train_path>\
            --test_noisy <test_noisy >\
            --test_clean <test_clean > \
            --writer <log_path> \
            --model BLSTM_05 \
            --task <denoise / synthesis>
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
