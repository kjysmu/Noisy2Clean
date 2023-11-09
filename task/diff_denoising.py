import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchaudio
import torch.nn.functional as F
import os

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    x_start: x0 (B, T, F) --> (B, 8, L)
    t: timestep information (B,)
    """    
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]# extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    # boardcasting into correct shape
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None].to(x_start.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None].to(x_start.device)

    # scale down the input, and scale up the noise as time increases?
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract_x0(x_t, epsilon, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    x_t: The output from q_sample
    epsilon: The noise predicted from the model
    t: timestep information
    """
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t] # extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_t.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_t.device)    

    # obtaining x0 based on the inverse of eq.4 of DDPM paper
    return (x_t - sqrt_one_minus_alphas_cumprod_t * epsilon) / sqrt_alphas_cumprod_t

def calculate_sdr(ref, est):
    """
    ref: (B, L)
    est: (B, L)
    """
    assert ref.dim()==est.dim(), f"ref {ref.shape} has a different size than est {est.shape}"
    
    s_true = ref
    s_artif = est - ref

    sdr = 10. * (
        torch.log10(torch.clip(torch.mean(s_true ** 2, -1), 1e-8, torch.inf)) \
        - torch.log10(torch.clip(torch.mean(s_artif ** 2, -1), 1e-8, torch.inf)))
    return sdr

class DiffSeparation(pl.LightningModule):

    def __init__(self, **task_args):
        super().__init__()
        self.save_hyperparameters()
        self.betas = linear_beta_schedule(self.hparams.beta_start,
                                          self.hparams.beta_end,
                                          timesteps=self.hparams.timesteps)
        
        # define alphas
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))        

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
             
        
    def step(self, batch):
        # batch["frame"] (B, 640, 88)
        # batch["audio"] (B, L)
        
        batch_size = batch.shape[0]
        
        if self.training:
            waveform = batch.sum(dim=1)  # (B, 2channel, L)
            source_label = batch # target (B, 4, 2, L)
            source_label = source_label.flatten(1,2)  
        else:
            waveform = batch[:,0]
            source_label = batch[:,1:]
            source_label = source_label.flatten(1,2)              
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ## sampling the same t within each batch, might not work well
        # t = torch.randint(0, self.hparams.timesteps, (1,), device=device)[0].long() # [0] to remove dimension
        # t_tensor = t.repeat(batch_size).to(roll.device)
        t = torch.randint(0,
                          self.hparams.timesteps,
                          (batch_size,),
                          device=self.device
                         ).long() # more diverse sampling
        
        noise = torch.randn_like(source_label) # creating label noise
        
        x_t = q_sample( # sampling noise at time t
            x_start=source_label,
            t=t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise)
        
        # feedforward
        source_pred = self(x_t, waveform, t) # predict the noise N(0, 1)
        diffusion_loss = self.p_losses(source_label, source_pred, loss_type=self.hparams.loss_type)

        
        # pred_roll = torch.sigmoid(pred_roll) # to convert logit into probability
        # amt_loss = F.binary_cross_entropy(pred_roll, roll)
        

        
        tensors = {
            "source_pred": source_pred,
            "source_label": source_label,
        }
              

        return diffusion_loss, tensors
    
    def p_losses(self, label, prediction, loss_type="l1"):

        if loss_type == 'l1':
            loss = F.l1_loss(label, prediction)
        elif loss_type == 'l2':
            loss = F.mse_loss(label, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(label, prediction)
        else:
            raise NotImplementedError()

        return loss    

    def training_step(self, batch, batch_idx):
        # batch.shape = (B, 4, 2, L)
        diff_loss, pred_dict = self.step(batch) # (batch, 8, len)
        label = batch # (batch, 4, 2, len)
        
        loss_wav = torch.nn.functional.mse_loss(
            pred_dict['source_pred'],
            pred_dict['source_label']
            )
        
        # label.shape = (batch, 4, 2, len)
        sdr = calculate_sdr(pred_dict['source_label'], pred_dict['source_pred'])
        sdr1, sdr2, sdr3, sdr4 = \
            torch.split(sdr,2, dim=1)
        
        self.log('Train/mse_wav', loss_wav)
        self.log('Train/diff_loss', diff_loss)
        self.log('Train/sdr', sdr.mean())
        self.log('Train/sdr1', sdr1.mean())
        self.log('Train/sdr2', sdr2.mean())
        self.log('Train/sdr3', sdr3.mean())
        self.log('Train/sdr4', sdr4.mean())
        
        return loss_wav + diff_loss
    
    def validation_step(self, batch, batch_idx):
        # batch.shape = (B, 4, 2, L)
        diff_loss, pred_dict = self.step(batch) # (batch, 8, len)
        label = batch[0,1:] # (batch, 4, 2, len)
        
        loss_wav = torch.nn.functional.mse_loss(
            pred_dict['source_pred'],
            pred_dict['source_label']
            )
        
        # label.shape = (batch, 4, 2, len)
        sdr = calculate_sdr(pred_dict['source_label'], pred_dict['source_pred'])
        sdr1, sdr2, sdr3, sdr4 = \
            torch.split(sdr,2, dim=1)
        
        self.log('Val/mse_wav', loss_wav)
        self.log('Val/diff_loss', diff_loss)
        self.log('Val/sdr', sdr.mean())
        self.log('Val/sdr1', sdr1.mean())
        self.log('Val/sdr2', sdr2.mean())
        self.log('Val/sdr3', sdr3.mean())
        self.log('Val/sdr4', sdr4.mean())
        return loss_wav + diff_loss, sdr, sdr1, sdr2, sdr3, sdr4

    

    def test_step(self, batch, batch_idx):
        # batch.shape = (B, 4, 2, L)
        diff_loss, pred_dict = self.step(batch) # (batch, 8, len)
        label = batch[0,1:] # (batch, 4, 2, len)
        
        if batch_idx==4:
            torchaudio.save(os.path.join(os.getcwd(),f"mix.mp3"),
                            batch[0,0].cpu(),
                            16000) 
            for idx, stem in enumerate(pred_dict['source_pred'][0].view(4,2,-1)):
                torchaudio.save(os.path.join(os.getcwd(),f"stem{idx+1}.mp3"),
                                stem.cpu(),
                                16000)   
        
        loss_wav = torch.nn.functional.mse_loss(
            pred_dict['source_pred'],
            pred_dict['source_label']
            )
        
        # label.shape = (batch, 4, 2, len)
        sdr = calculate_sdr(pred_dict['source_label'], pred_dict['source_pred'])
        sdr1, sdr2, sdr3, sdr4 = \
            torch.split(sdr,2, dim=1)
        
        self.log('Test/mse_wav', loss_wav)
        self.log('Test/diff_loss', diff_loss)
        self.log('Test/sdr', sdr.mean())
        self.log('Test/sdr1', sdr1.mean())
        self.log('Test/sdr2', sdr2.mean())
        self.log('Test/sdr3', sdr3.mean())
        self.log('Test/sdr4', sdr4.mean())
        return loss_wav + diff_loss, sdr, sdr1, sdr2, sdr3, sdr4

        

    def configure_optimizers(self):
        r"""Configure optimizer."""
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
