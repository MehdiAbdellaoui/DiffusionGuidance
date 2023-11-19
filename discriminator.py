from guided_diffusion.script_util import create_classifier
import torch
import math
import numpy as np



def load_discriminator(ckpt, device, grads=True, conditioned=True):
    """
    Examples:
        evaluate = load_discriminator('models/discriminator_epoch99.pt', device, conditioned=True)
        evaluate(noisy_images, t, cond)

    Args:
        ckpt: Path to discriminator checkpoint
        device: Choose to load model to CPU or GPU
        grads: Choose whether torch grads should be active during evaluation
        conditioned: Whether the checkpoint loads a conditioned model or not

    Returns: A discriminator evaluation function

    """
    # Load the complete discriminator and return the evaluation function
    adm_model = get_ADM_model().to(device)
    discriminator = get_discriminator_model(conditioned, ckpt).to(device)

    def evaluate(x, t, cond=None):
        with torch.enable_grad() if grads else torch.no_grad():
            feature_x = adm_model(x, t, cond, features=True, sigmoid=False)
            output = discriminator(feature_x, t, cond, features=False, sigmoid=True)[:, 0]
        return output

    return evaluate


# Create the ADM model as explained in the paper and the given ADM checkpoint
def get_ADM_model():
    adm_classifier = create_classifier(
        image_size=32,
        classifier_in_channels=3,
        classifier_out_channels=1000,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="32,16,8",
        classifier_pool='attention',
        conditioned=False
    )
    pretrained_adm = torch.load('guided_diffusion/model/32x32_classifier.pt')
    adm_classifier.load_state_dict(pretrained_adm)
    return adm_classifier


# Create the discriminator as given in the paper
def get_discriminator_model(conditioned, ckpt=None):
    discriminator = create_classifier(
        image_size=8,
        classifier_in_channels=512,
        classifier_out_channels=1,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="8",
        classifier_pool='attention',
        conditioned=conditioned
    )
    if ckpt is not None:
        pretrained_discriminator = torch.load(ckpt)
        discriminator.load_state_dict(pretrained_discriminator)
    return discriminator

def get_gradient_density_ratio(discriminator, vpsde, input_, std_wve_t, time_mid, time_max, img_size, class_labels):
    
    # merging two checkpoints from different diffusion strategies (page 23)
    mean_tau, tau = vpsde.transform_WVE_to_LVP(std_wve_t)

    # outside range of application of DG
    if tau.min() > time_max or tau.min() < time_mid or discriminator == None:
        return torch.zeros_like(input_), torch.ones(input_.shape[0], device=input_.device) * 1e9  
   
    input_ = mean_tau.reshape(input_.shape) * input_

    with torch.enable_grad():
        # add gradient to input 
        x_ = torch.tensor(input_, dtype=torch.float64, requires_grad=True)
        
        # classifier checkpoints are trained with Linear VP for 32x32 and Cosine VP for 64x64 (page 23)
        if img_size == 64:
            tau = vpsde.get_cosine_time_from_linear_time(tau)

        tau = torch.ones(input_.shape[0], device=tau.device) * tau 
        
        density_log_ratio = get_density_log_ratio(discriminator, x_, tau, class_labels)

        discriminator_score = torch.autograd.grad(outputs=density_log_ratio.sum(), inputs=x_, retain_graph=False)[0]
    
        discriminator_score *= - ((std_wve_t.reshape(discriminator_score.shape) ** 2) * mean_tau.reshape(discriminator_score.shape))
        
        return discriminator_score, density_log_ratio

def get_density_log_ratio(discriminator, x_, tau, class_labels):
    
    logits = discriminator(x_, tau, class_labels)

    # clip range for sampling according to Table 8 (paper 21)
    prediction = torch.clip(logits, 1e-5, 1. - 1e-5)  

    density_log_ratio = torch.log(prediction / (1. - prediction))
    
    return density_log_ratio

# Implemented based on page 24 and 15
class WVEtoLVP:
    """Class to convert between the training modes of EDM and ADM"""
    def __init__(self):
        self.beta_min = 0.1
        self.beta_max = 20.
        self.T = 1.     # For u sampled uniformly from [0, T]

    def transform_to_tau(self, var_wve_t):
        # See page 24 for derivation and explanation
        temp = self.beta_min ** 2 + 2. * (self.beta_max - self.beta_min) * torch.log(1. + var_wve_t)
        tau = (-self.beta_min + torch.sqrt(temp)) / (self.beta_max - self.beta_min)
        return tau

    def marginal_prob(self, t):
        mean_coef = -t ** 2 * (self.beta_max - self.beta_min) / 4 - t * self.beta_min / 2
        mean = torch.exp(mean_coef)
        std = torch.sqrt(1. - torch.exp(2. * mean_coef))
        return mean, std

    def transform_WVE_to_LVP(self, std_wve_t):
        tau = self.transform_to_tau(std_wve_t ** 2)
        mean, _ = self.marginal_prob(tau)
        return mean, tau

    def antiderivative(self, t):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        integral_beta = 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min
        return torch.log(1. - torch.exp(- integral_beta)) + integral_beta

    def generate_diffusion_times(self, n_samples, device, t_min=1e-5):
        # See page 15 for derivations of importance sampler!
        # Calculate F(t) = u * Z - F(t_min), where F(t) is the antiderivative of the importance weight
        anti_t_min = self.antiderivative(t_min)
        Z = self.antiderivative(self.T) - anti_t_min
        u = torch.rand(n_samples, device=device) * Z + anti_t_min
        return self.transform_to_tau(torch.exp(u))
    
    def get_cosine_time_from_linear_time(self, linear_time):
        
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models": https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L39
        s = 0.008
        cosine_schedule = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        
        sqrt_alpha_t_bar = torch.exp(-0.25 * linear_time ** 2 * (self.beta_1 - self.beta_0) - 0.5 * linear_time * self.beta_0)
        time = torch.arccos(np.sqrt(cosine_schedule(0)) * sqrt_alpha_t_bar)
        cosine_time = self.T * ((1. + s) * 2. / np.pi * time - s)
        
        return cosine_time

if __name__ == '__main__':
    get_ADM_model()
    get_discriminator_model(True)
