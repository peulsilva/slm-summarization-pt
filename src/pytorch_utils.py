def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,} ")
    print(f"Trainable Parameters: {trainable_params:,} ({round(trainable_params/total_params, 2)*100} %)")