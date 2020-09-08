import os, argparse, torch, random
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pandas as pd
import pdb, sys

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True


# data path
Train_noisy_path = {
    'ori'       :'../EMMA_data_pt_mel/train/Noisy',
#     'emma'      :'../EMMA_data_pt_mel/train/Noisy_emma/',
    'emma'      :'../EMMA_data_pt_mel/train/clean',
}
Train_clean_path = '../EMMA_data_pt_mel/train/clean'
Test_path = {
#     'noisy':'../EMMA_data/test/Noisy',
    'noisy':'../EMMA_data/test/clean',
    'clean':'../EMMA_data/test/clean'
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_path', type=str, default='../denoise_pt/train')
    parser.add_argument('--test_noisy', type=str, default='../../EMMA/EMMA_data/test/Noisy')
    parser.add_argument('--test_clean', type=str, default='../../EMMA/EMMA_data/test/clean')
    parser.add_argument('--pwg_path', type=str, default='./pretrain_model')
    parser.add_argument('--writer', type=str, default='/data1/user_yuwen/logs')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=4)  
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--encode_loss' , action='store_true', default=False)
    parser.add_argument('--model', type=str, default='DDAE_13') 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume' , action='store_true', default=False)
    parser.add_argument('--task', type=str, default='denoise')  # denoise / synthesis
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
    
    
    # get parameter
    args = get_args()

    # declair path
    checkpoint_path = f'./checkpoint/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    model_path = f'./save_model/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    score_path = f'./Result/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.csv'
    
    # tensorboard
    writer = SummaryWriter(args.writer)

    exec (f"from model.{args.model.split('_')[0]} import {args.model} as model")
    model     = model()
#     if args.update=='all':
#         param=model
#     else:
#         exec( f"param = model.{args.update}")
    model, epoch, best_loss, optimizer, criterion, device = Load_model(args,model,checkpoint_path,model)
    
    loader = Load_data(args) if args.mode == 'train' else 0

    Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, writer, model_path, score_path,args)
    try:
        if args.mode == 'train':
            Trainer.train()
        Trainer.test()
        
        
        
        
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }
        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
