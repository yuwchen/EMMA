# '''
# Here is the script of denoise and sybthesis command. for example, if the folder path are like the following condition:

###################### folder example #########################

# train noisy audio folder     : ../../EMMA/EMMA_data/train/Noisy
# train clean audio folder     : ../../EMMA/EMMA_data/train/clean
# test noisy audio folder      : ../../EMMA/EMMA_data/test/Noisy
# test clean audio folder      : ../../EMMA/EMMA_data/test/clean
# denoise train folder         : ../denoise_pt/train
# synthesis train folder       : ../synthesis_pt/train
# pretrain para-Wavegan        : ./pretrain_model
# log folder                   : ./logs

###################### model example #########################

# ==== denoise ======
# audio only           : DDAE_01, BLSTM_01
# directly concatenate : DDAE_02, BLSTM_02
# EMMA encoder         : DDAE_03, BLSTM_03
# EMMA+spec encoder    : DDAE_04, BLSTM_04


# ==== synthesis ====
# spec only            : BLSTM_05
# spec with encode     : BLSTM_06

#################################################################


# ====================            denoise            ===================
# step 1 preparing training data
python gen_pt.py --noisy_path ../../EMMA/EMMA_data/train/Noisy \
                --clean_path ../../EMMA/EMMA_data/train/clean \
                --out_path ../denoise_pt/train \
                --pwg_path ./pretrain_model \
                --task denoise

# step 2 training mode
export CUDA_VISIBLE_DEVICES='1'
python main.py --mode test \
            --train_path ../denoise_pt/train \
            --test_noisy ../../EMMA/EMMA_data/test/Noisy \
            --test_clean ../../EMMA/EMMA_data/test/clean \
            --writer ./logs \
            --epochs 3 \
            --batch_size 128 \
            --lr 0.0001 \
            --loss_fn l1 \
            --optim adam \
            --model BLSTM_60 \
            --task denoise




# ====================            synthesis          ===================
# step 1 preparing training data
python gen_pt.py --clean_path ../../EMMA/EMMA_data/train/clean \
                --out_path ../synthesis_pt/train \
                --pwg_path ./pretrain_model \
                --task synthesis

# step 2-1 training mode without encode loss
export CUDA_VISIBLE_DEVICES='1'
python main.py --mode train \
            --train_path ../synthesis_pt/train \
            --test_noisy ../../EMMA/EMMA_data/test/Noisy \
            --test_clean ../../EMMA/EMMA_data/test/clean \
            --writer ./logs \
            --epochs 250 \
            --batch_size 4 \
            --lr 0.0001 \
            --loss_fn l1 \
            --optim adam \
            --model BLSTM_05 \
            --task synthesis
            
# step 2-2 training mode with encode loss            
python main.py --mode train \
            --train_path ../synthesis_pt/train \
            --test_noisy ../../EMMA/EMMA_data/test/Noisy \
            --test_clean ../../EMMA/EMMA_data/test/clean \
            --writer ./logs \
            --epochs 250 \
            --batch_size 4 \
            --lr 0.0001 \
            --loss_fn l1 \
            --optim adam \
            --encode_loss \
            --model BLSTM_06 \
            --task synthesis