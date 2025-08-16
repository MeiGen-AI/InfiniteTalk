import torch

sigma_fn = lambda t: t.neg().exp()
t_fn = lambda sigma: sigma.log().neg()
phi1_fn = lambda t: torch.expm1(t) / t
phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

class FlowMatchSchedulerResMultistep():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, extra_one_step=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)
        self.prev_model_output = None
        self.old_sigma_next = None

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0):
        #Generate the full sigma schedule (from max to min)
        if self.extra_one_step:
            sigma_start = self.sigma_min + \
                (self.sigma_max - self.sigma_min) * denoising_strength
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        full_sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_train_timesteps)
        ss = len(full_sigmas) / num_inference_steps
        sigmas = []
        for x in range(num_inference_steps):
            idx = int(round(x * ss))
            sigmas.append(float(full_sigmas[idx]))
        sigmas.append(0.0)
        self.sigmas = torch.FloatTensor(sigmas)
        self.sigmas = self.shift * self.sigmas / \
             (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        #print(f"Timesteps: {self.timesteps}, Sigmas: {self.sigmas}")


    def step(self, model_output, timestep, sample):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        if timestep.ndim == 0:
            timestep_id = torch.argmin((self.timesteps - timestep).abs(), dim=0)
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sigma_prev = self.sigmas[timestep_id - 1].reshape(-1, 1, 1, 1) if timestep_id > 0 else sigma
        if (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_next = torch.tensor(0)
        else:
            sigma_next = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
      
        x0_pred = (sample - sigma * model_output)

        if sigma_next == 0 or self.prev_model_output is None:
            x = sample + model_output * (sigma_next - sigma)
        else:
            t, t_old, t_next, t_prev = t_fn(sigma), t_fn(self.old_sigma_next), t_fn(sigma_next), t_fn(sigma_prev)
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            x = sigma_fn(h) * sample + h * (b1 * x0_pred + b2 * self.prev_model_output)

        self.old_sigma_next = sigma_next
        self.prev_model_output = x0_pred
        return x
        

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

