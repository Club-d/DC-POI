import os.path
import torch
import gol
import time
from dataset import getDatasets, collate_edge
from torch.utils.data import DataLoader
from model import DiffPOI
from evaluation import eval_model
from pprint import pformat


def train_eval(model: DiffPOI, _all_ds):
    trn_set, val_set, tst_set = _all_ds
    trn_loader = DataLoader(trn_set, batch_size=gol.BATCH_SZ, shuffle=True, collate_fn=collate_edge)
    opt = torch.optim.AdamW(model.parameters(), lr=gol.conf['lr'], weight_decay=gol.conf['decay'])
    batch_num = len(trn_set) // gol.BATCH_SZ
    best_val_epoch, best_val_ndcg, best_val_recall = 0, 0., 0.
    best_tst_epoch, best_tst_ndcg, best_tst_recall = 0, 0., 0.
    ave_tot, ave_rec, ave_fis = 0., 0., 0.
    tst_result = None

    for epoch in range(gol.EPOCH):
        start = time.time()
        model.train()
        for idx, batch in enumerate(trn_loader):
            rec_loss, fisher_loss = model.getTrainLoss(batch)
            tot_loss = rec_loss +  gol.conf['beta'] * fisher_loss  
            ave_tot += tot_loss.item()
            ave_rec += rec_loss.item()
            opt.zero_grad()
            tot_loss.backward()
            opt.step()
            if idx % (batch_num // 5) == 0:
                gol.pLog(f'Batch {idx} / {batch_num}, Loss: {tot_loss.item():.5f}' +
                         f' = CE Loss: {rec_loss.item():.5f} + Fisher Loss: {fisher_loss.item():.5f}')

        ave_tot /= batch_num + 1
        ave_rec /= batch_num + 1
       
        end = time.time()
        consume = end - start
        gol.pLog(f'Epoch {epoch} / {gol.EPOCH}, Loss: {ave_tot:.5f}')
        gol.pLog(f'Training time consuming in this epoch is: {consume:.2f} seconds.\n')

        val_results, _ = eval_model(model, val_set)
        gol.pLog(f'Valid NDCG@5: {val_results["ndcg"][2]:.5f}, Recall@2: {val_results["recall"][1]:.5f}, Recall@5: {val_results["recall"][2]:.5f}')

        if val_results["recall"][2] > best_val_recall or epoch == 0:
            best_val_epoch, best_val_ndcg, best_val_recall = epoch, val_results["ndcg"][2], val_results["recall"][2]
            tst_results, _ = eval_model(model, tst_set)
            if tst_results["recall"][2] > best_tst_recall or epoch == 0:
                best_tst_epoch, best_tst_ndcg, best_tst_recall = epoch, tst_results["ndcg"][2], tst_results["recall"][2]
                best_results = tst_results
            gol.pLog(f'New test result:\n {pformat(tst_results)}')
            if gol.SAVE:
                torch.save(model.cpu().state_dict(), w_path)
                model.to(gol.device)
        
        if epoch - best_tst_epoch == gol.patience:
            gol.pLog(f'Stop training after {gol.patience} epochs without test improvement.')
            break
        
        gol.pLog(f'Test NDCG@5: {tst_results["ndcg"][2]:.5f}, Recall@2: {tst_results["recall"][1]:.5f}, Recall@5: {tst_results["recall"][2]:.5f}')
        gol.pLog(f'Best test Recall@5 at epoch {best_tst_epoch}\n')

    

    return best_results, best_tst_epoch

if __name__ == '__main__':
    print("\nAccelerating Dual-Core Diffussion Model for Recommendation on Point of Interest(DC-POI+).\n")
    print("Loading...")

    w_path = os.path.join(gol.FILE_PATH, 'weight.pth')
    n_user, n_poi, n_cat, all_ds, geo_graph = getDatasets(gol.DATA_PATH, gol.dataset.lower())

    recModel = DiffPOI(n_user, n_poi, n_cat, geo_graph)

    if gol.LOAD:
        recModel.load_state_dict(torch.load(w_path))
    recModel = recModel.to(gol.device)

    gol.pLog(f'Start Training')
    gol.pLog(f'Dropout: {1 - gol.conf["keepprob"] if gol.conf["dropout"] else 0}\n')

    best_results, best_epoch = train_eval(recModel, all_ds)
    gol.pLog(f'Training on {gol.dataset.upper()} finished, best test Recall@5 at epoch {best_epoch}')
    gol.pLog(f'Best result:\n{pformat(best_results)}')

    print("Done.\n")
