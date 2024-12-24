import pytorch_lightning as pl
from torch.utils.data import DataLoader

def test_model(model, test_dataset, batch_size):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    trainer = pl.Trainer(gpus=1)
    trainer.test(model, test_loader)

