import torch
import wandb
import copy

def train_autoenc(
    autoencoder, 
    train_features,
    validation_features, 
    train_corerlation,
    validation_correlation,
    lr_init=1e-3, 
    lr_decay=0.3, 
    patience=10, 
    min_lr=1e-5, 
    wandb_project=None,
    wandb_config=None,
    wandb_name="", 
    log_every_n=100,
    device='cpu',
):
    autoencoder.to(device)
    train_features = train_features.to(device)
    validation_features = validation_features.to(device)

    if wandb_project:
        wandb.init(
            project=wandb_project,
            config=wandb_config or {},
        )
        wandb.run.name = wandb_name

    optim = torch.optim.Adam(autoencoder.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=lr_decay, patience=patience)

    epochs = int(1e15)

    best_score = -float("inf")   # because higher correlation is better
    best_state_dict = None

    for epoch in range(epochs):
        autoencoder.train()
        recon = autoencoder(train_features)
        train_loss = (train_features - recon).pow(2).sum(dim=1).mean()

        optim.zero_grad()
        train_loss.backward()
        optim.step()

        if epoch % log_every_n == 0:
            autoencoder.eval()
            with torch.no_grad():
                recon = autoencoder(validation_features)
                validation_loss = (validation_features - recon).pow(2).sum(dim=1).mean()
            
            train_score = train_corerlation()
            validation_score = validation_correlation()

            scheduler.step(validation_score)
            
            if validation_score > best_score:
                best_state_dict = copy.deepcopy(autoencoder.state_dict())

            lr = optim.param_groups[0]['lr']
            if lr < min_lr:
                break

            log_dict = {
                'training recon loss': train_loss.item(),
                'validation recon loss': validation_loss.item(), 
                'training correlation': train_score, 
                'validation correlation': validation_score,
                'learing rate': lr,
            }    
            wandb.log(log_dict)

    autoencoder.load_state_dict(best_state_dict)

    if wandb_project:
        wandb.finish()