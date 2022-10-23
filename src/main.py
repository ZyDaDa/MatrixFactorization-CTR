from tqdm import tqdm
from dataset import load_data
import torch
from parse import get_parse
from utils import fix_seed, metrics
from model import MF    
import numpy as np

def main():
    fix_seed()
    args = get_parse()

    train_loader, test_loader, user_num, item_num = load_data(args)

    model = MF(args, user_num, item_num)
    model.to(args.device)

    # trainner
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)

    for e in range(args.epoch):
        model.train()
        all_loss = 0.0
        bar = tqdm(train_loader, total=len(train_loader),ncols=100)
        for batch in bar:
            scores = model(dict([(k,v.to(args.device)) if torch.is_tensor(v) else(k,v) for k,v in batch.items()]))
            optimizer.zero_grad() 
            loss = model.loss_function(scores, batch['label'].to(args.device)) 
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            bar.set_postfix(Epoch=e, LR=optimizer.param_groups[0]['lr'], Train_Loss=loss.item()/batch['label'].size(0))
        print('epoch%d - loss%f'%(e,all_loss/len(train_loader)))
        
        model.eval()
        y_true = []
        y_pre = []
        for batch in tqdm(test_loader,ncols=80,desc='test'):
            scores = model(dict([(k,v.to(args.device)) if torch.is_tensor(v) else(k,v) for k,v in batch.items()]))
            loss = model.loss_function(scores, batch['label'].to(args.device)) 

            y_true.append(batch['label'].numpy())
            y_pre.append(scores.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pre = np.concatenate(y_pre)
        results = metrics(y_true, y_pre)

        for k, v in results.items():
            print("%s\t%.4f"%(k,v))

if __name__ == '__main__':

    main()

