import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from .ddim import DDIMSampler

class DDIMSampler_VD(DDIMSampler):
    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               xt=None,
               conditioning=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               xtype='image',
               ctype='prompt',
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling(
            shape,
            xt=xt,
            conditioning=conditioning, 
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            xtype=xtype,
            ctype=ctype,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, 
                      shape,
                      xt=None,
                      conditioning=None,
                      unconditional_guidance_scale=1., 
                      unconditional_conditioning=None,
                      xtype='image',
                      ctype='prompt',
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      noise_dropout=0., 
                      temperature=1., 
                      log_every_t=100,):

        device = self.model.model.diffusion_model.device
        bs = shape[0]
        if xt is None:
            xt = torch.randn(shape, device=device, dtype=conditioning.dtype)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        pred_xt = xt
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                pred_xt, conditioning, ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning, 
                xtype=xtype,
                ctype=ctype,
                use_original_steps=ddim_use_original_steps,
                noise_dropout=noise_dropout,
                temperature=temperature,)
            pred_xt, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, conditioning, t, index, 
                      unconditional_guidance_scale=1., 
                      unconditional_conditioning=None, 
                      xtype='image',
                      ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,):

        b, *_, device = *x.shape, self.model.model.diffusion_model.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, conditioning, xtype=xtype, ctype=ctype)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, conditioning])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, xtype=xtype, ctype=ctype).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample_dc(self,
               steps,
               shape,
               xt=None,
               first_conditioning=None,
               second_conditioning=None,
               unconditional_guidance_scale=1.,
               xtype='image',
               first_ctype='prompt',
               second_ctype='prompt',
               eta=0.,
               temperature=1.,
               mixed_ratio=0.5,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling_dc(
            shape,
            xt=xt,
            first_conditioning=first_conditioning,
            second_conditioning=second_conditioning,
            unconditional_guidance_scale=unconditional_guidance_scale,
            xtype=xtype,
            first_ctype=first_ctype,
            second_ctype=second_ctype,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,
            mixed_ratio=mixed_ratio, )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_dc(self, 
                      shape,
                      xt=None,
                      first_conditioning=None,
                      second_conditioning=None,
                      unconditional_guidance_scale=1., 
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      noise_dropout=0., 
                      temperature=1.,
                      mixed_ratio=0.5,
                      log_every_t=100,):

        device = self.model.model.diffusion_model.device
        bs = shape[0]
        if xt is None:
            xt = torch.randn(shape, device=device, dtype=first_conditioning[1].dtype)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        pred_xt = xt
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim_dc(
                pred_xt, 
                first_conditioning, 
                second_conditioning, 
                ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                xtype=xtype,
                first_ctype=first_ctype,
                second_ctype=second_ctype,
                use_original_steps=ddim_use_original_steps,
                noise_dropout=noise_dropout,
                temperature=temperature,
                mixed_ratio=mixed_ratio,)
            pred_xt, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim_dc(self, x, 
                      first_conditioning,
                      second_conditioning,
                      t, index, 
                      unconditional_guidance_scale=1., 
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,
                      mixed_ratio=0.5,):

        b, *_, device = *x.shape, self.model.model.diffusion_model.device

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        first_c = torch.cat(first_conditioning)
        second_c = torch.cat(second_conditioning)

        e_t_uncond, e_t = self.model.apply_model_dc(
            x_in, t_in, first_c, second_c, xtype=xtype, first_ctype=first_ctype, second_ctype=second_ctype, mixed_ratio=mixed_ratio).chunk(2)

        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
              unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
       num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

       assert t_enc <= num_reference_steps
       num_steps = t_enc

       if use_original_steps:
           alphas_next = self.alphas_cumprod[:num_steps]
           alphas = self.alphas_cumprod_prev[:num_steps]
       else:
           alphas_next = self.ddim_alphas[:num_steps]
           alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])
       
       alphas_next = alphas_next.to(x0.device)
       alphas = alphas.to(x0.device)
       x_next = x0
       intermediates = []
       inter_steps = []
       for i in tqdm(range(num_steps), desc='Encoding Image'):
           t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
           if unconditional_guidance_scale == 1.:
               noise_pred = self.model.apply_model(x_next, t, c)
           else:
               assert unconditional_conditioning is not None
               e_t_uncond, noise_pred = torch.chunk(
                   self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                          torch.cat((unconditional_conditioning, c))), 2)
               noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

           xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
           weighted_noise_pred = alphas_next[i].sqrt() * (
                   (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
           x_next = xt_weighted + weighted_noise_pred
           if return_intermediates and i % (
                   num_steps // return_intermediates) == 0 and i < num_steps - 1:
               intermediates.append(x_next)
               inter_steps.append(i)
           elif return_intermediates and i >= num_steps - 2:
               intermediates.append(x_next)
               inter_steps.append(i)
           if callback: callback(i)

       out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
       if return_intermediates:
           out.update({'intermediates': intermediates})
       return x_next, out
    
    
    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
            
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(t.device)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(t.device)

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None, xtype='image', ctype='vision',
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, xtype=xtype, ctype=ctype, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
    
    @torch.no_grad()
    def decode_dc(self, x_latent, first_conditioning, second_conditioning, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None, xtype='image', first_ctype='vision', second_ctype='prompt',
               use_original_steps=False, mixed_ratio=0.5, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim_dc(
                x_dec, 
                first_conditioning, 
                second_conditioning, 
                ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                xtype=xtype,
                first_ctype=first_ctype,
                second_ctype=second_ctype,
                use_original_steps=use_original_steps,
                noise_dropout=0,
                temperature=1,
                mixed_ratio=mixed_ratio,)
            if callback: callback(i)
        return x_dec
    
    
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))