import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

def train_model(model, train_loader, val_loader, epochs, gpus=1, output_dir='./best_model/'):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename='best_model'
    )
    trainer = pl.Trainer(
        max_epochs=epochs, 
        gpus=gpus, 
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=5)],
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer, checkpoint_callback
