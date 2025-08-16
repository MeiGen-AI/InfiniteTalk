import torch



class FlowMatchSchedulerPusa():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            # timestep = timestep.cpu()
            self.timesteps = self.timesteps.to(timestep.device)
            self.sigmas = self.sigmas.to(timestep.device)
            model_output = model_output.to(timestep.device)
            sample = sample.to(timestep.device)
        if len(timestep.shape) == 1:
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
            sigma = self.sigmas[timestep_id]
            if to_final or timestep_id + 1 >= len(self.timesteps):
                sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
            else:
                sigma_ = self.sigmas[timestep_id + 1]
            prev_sample = sample + model_output * (sigma_ - sigma)
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep).abs(), dim=0)
            sigma = self.sigmas[timestep_id].unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).to(sample.device)
            # Handle sigma_ calculation for each timestep_id element
            if to_final or torch.any(timestep_id + 1 >= len(self.timesteps)):
                default_value = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
                # Create sigma_ with the same dtype as self.sigmas
                sigma_ = torch.ones_like(timestep_id, dtype=self.sigmas.dtype, device=sample.device) * default_value
                valid_indices = timestep_id + 1 < len(self.timesteps)
                if torch.any(valid_indices):
                    # Convert indices to the appropriate type for indexing
                    valid_timestep_ids = timestep_id[valid_indices] 
                    sigma_[valid_indices] = self.sigmas[(valid_timestep_ids + 1).to(torch.long)]
            else:
                sigma_ = self.sigmas[(timestep_id + 1).to(torch.long)]
                
            
            # Reshape sigma_ to match sigma's dimensions for the operation
            sigma_ = sigma_.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).to(sample.device)
            if torch.any(timestep == 0):
                zero_indices = torch.where(timestep == 0)[1].to(torch.long)
                sigma[:,:,zero_indices] = 0
            #print("sigma", sigma[0,0,:,0,0], '\n', "sigma_", sigma_[0,0,:,0,0])
            
            prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            # timestep = timestep.cpu()
            self.timesteps = self.timesteps.to(timestep.device)
            self.sigmas = self.sigmas.to(timestep.device)
        if len(timestep.shape) == 1:
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
            sigma = self.sigmas[timestep_id]
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep).abs(), dim=0)
            sigma = self.sigmas[timestep_id].unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).to(sample.device)
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            # timestep = timestep.cpu()
            self.timesteps = self.timesteps.to(timestep.device)
            self.sigmas = self.sigmas.to(timestep.device)
        if len(timestep.shape) == 1:
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
            sigma = self.sigmas[timestep_id]
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep).abs(), dim=0)
            sigma = self.sigmas[timestep_id].unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).to(original_samples.device)
        sample = (1 - sigma) * original_samples + sigma * noise
        
        return sample
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        if isinstance(timestep, torch.Tensor):
            self.timesteps = self.timesteps.to(timestep.device)
            self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        if len(timestep.shape) == 1:
            timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.to(self.timesteps.device)).abs(), dim=0) 
        weights = self.linear_timesteps_weights[timestep_id].to(self.timesteps.device)
        return weights