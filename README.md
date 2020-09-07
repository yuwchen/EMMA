# EMMA
### Requirement
### Usage

* Step 1 training data \
```
python gen_pt.py --noisy_path < noisy_path (only needed in denoise mode )> \   
    --clean_path < clean_path > \       
    --out_path <out_path> \       
    --pwg_path <parallel_wavegan_path> \       
    --task < denoise / synthesis>
```

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
* [Bio-ASP Lab](/https://bio-asplab.citi.sinica.edu.tw/), CITI, Academia Sinica, Taipei, Taiwan
