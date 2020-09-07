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

### Step 1 - Prepare training data 

Transform waveform to spectrogram by STFT, and divide spectrogram by frame size 64. 

Setup yout dataset to be the following structure.

```
/path/to/clean/
├── utt_1.wav
├── utt_1.mat
│   ...
│   ...
├── utt_N.wav
└── utt_N.mat
```
where ***.wav** is the clean audio file and ***.mat** is the corresponding EMMA data


* **Speech enhancement**

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
python gen_pt.py --noisy_path </path/to/noisy/data/> \   
    --clean_path </path/to/clean/data/> \       
    --out_path </path/to/training/data/> \       
    --task denoise
```

* **Speech synthesis**

Generate the training data for speech synthesis.  

```
python gen_pt.py --clean_path </path/to/clean/data/> \       
    --out_path </ptah/to/output/data/> \       
    --pwg_path </path/to/parallel_wavegan/> \       
    --task synthesis
```

  
### Step 2 - Train your own model

* **Speech enhancement**


* **Speech synthesis**

simple EMMA-to-speech model
```
python main.py --mode train \
            --train_path </path/to/training/data/>\
            --writer </path/to/log/file/> \
            --model BLSTM_05 \
            --task synthesis
            
```

audio-guided EMMA-to-speech model
```
python main.py --mode train \
            --train_path </path/to/training/data/>\
            --writer </path/to/log/file/> \
            --model BLSTM_06 \
            --encode_loss \
            --task synthesis
            
```



### Step 3 - Test the results
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
