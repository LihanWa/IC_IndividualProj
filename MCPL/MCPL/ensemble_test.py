from pytorch_lightning import Trainer
import torch

def ensemble_test(checkpoint_paths, datamodule):
    """多模型集成测试"""
    models = []
    for ckpt_path in checkpoint_paths:
        model = ClassificationMGCA.load_from_checkpoint(ckpt_path, strict=False)
        model.eval()
        models.append(model)
    
    trainer = Trainer(accelerator="gpu", devices=1, logger=False)
    
    # 收集所有模型的预测
    all_predictions = []
    for model in models:
        test_results = trainer.predict(model, datamodule)
        predictions = torch.cat([batch for batch in test_results])
        all_predictions.append(predictions)
    
    # 平均集成
    ensemble_predictions = torch.stack(all_predictions).mean(dim=0)
    
    return ensemble_predictions 