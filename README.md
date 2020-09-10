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


Setup your dataset to be the following structure.

```
/path/to/clean
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
/path/to/noisy/data
├── SNR_1
│   ├── noise_type1
│.  │.  ├──utt_1.wav
├── SNR_2
│   ├── noise_type2
│.  │.  ├──utt_1.wav
│   ...
└── SNR_N
    ├── noise_typeN
    │.  ├──utt_1.wav    
    
```

Use gen_pt.py to transform waveform to spectrogram by STFT, and divide spectrogram by frame size 64. 

  
Generate the training data for speech enhancement.  
```
python gen_pt.py --noisy_path </path/to/noisy/data> \   
                 --clean_path </path/to/clean/data> \       
                 --out_path </path/to/training/data> \       
                 --task denoise
```

* **Speech synthesis**

Generate the training data for speech synthesis.  
```
python gen_pt.py --clean_path </path/to/clean/data> \       
                 --out_path </ptah/to/output/data> \       
                 --task synthesis
```

  
### Step 2 - Train your own model

* **Speech enhancement**

```
python main.py --mode train \
               --train_path </path/to/training/data> \
               --writer </path/to/logs> \
               --model <Model_name> \
               --task denoise

```           
**Model_name**

| Method                                            | TDNN          | BLSTM         |
| ------------------------------------------------- |:-------------:|:-------------:|
| Audio only                                        | DDAE_01       | BLSTM_01      |
| Direct concatenate                                | DDAE_02       | BLSTM_02      |
| Unilateral encoding (EMMA encoder)                | DDAE_03       | BLSTM_03      |
| Bilateral encoding (EMMA encoder & audio encoder) | DDAE_04       | BLSTM_04      |




* **Speech synthesis**

simple EMMA-to-speech model
```
python main.py --mode train \
               --train_path </path/to/training/data>\
               --writer </path/to/log/file> \
               --model BLSTM_05 \
               --task synthesis
            
```

audio-guided EMMA-to-speech model
```
python main.py --mode train \
               --train_path </path/to/training/data>\
               --writer </path/to/log/file> \
               --model BLSTM_06 \
               --encode_loss \ 
               --task synthesis
            
```


### Step 3 - Test the results

* **Speech enhancement**

```
python main.py --mode test \
               --test_noisy </path/to/noisy> \
               --test_clean </path/to/clean> \
               --model <Model_name> \
               --task denoise      
```


* **Speech synthesis**


simple EMMA-to-speech model
```
python main.py --mode test \
               --test_clean </path/to/clean> \
               --model BLSTM_05 \
               --task synthesis       
```


audio-guided EMMA-to-speech model
```
python main.py --mode test \
               --test_clean </path/to/clean> \
               --encode_loss \ 
               --model BLSTM_06 \
               --task synthesis       
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
