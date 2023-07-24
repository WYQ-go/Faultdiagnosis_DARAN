import torch


class DataConfig:
    seed = 32210
    torch_seed = 91313

    class_num = 10
    sampleNum = 400
    sampleLen = 800
    test_size = 0.5
    share_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [0, 1, 4, 6, 7]
    # [0, 1, 3, 8]
    # [0, 2, 5]

    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [0, 3, 4, 6, 7, 8, 9]
    # [0, 2, 4, 8]
    # [0, 3, 5]

    optimizer = 'Adam'
    optim_hparas = {
        'lr': 1e-3,
        'weight_decay': 1e-4
    }
    lbd = 1.0

    batch_size = 512
    n_epoches = 3000
    early_stopping = 1000

    use_lambda_scheduler = True
    resume = False
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    root = 'D:\Projects\ssl_fault_diagnosis/CWRU_master'
    path = root + '/' + 'dataset'


    train_path = path + '/' + "1772FE_timesource_train.csv"
    test_path = path + '/' + "1772DE_timetarget_train.csv"
    Sfile_path = path + '/' + "1772FE_timesource_train.csv"
    STE_path = path + '/' + "1772FE_timesource_test.csv"
    TTRfile_path = path + '/' + "1772DE_timetarget_train.csv"
    TTEfile_path = path + '/' + "1772DE_timetarget_test.csv"
