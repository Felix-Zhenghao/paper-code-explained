import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.base import _expand_token
from modules.latent_distillation_modules import Encoder, Decoder

class AdaptiveLengthImageTokenizer(nn.Module):
    def __init__(self, 
            base_tokenizer,
            encoder_width, encoder_num_layers, encoder_num_heads,
            decoder_width, decoder_num_layers, decoder_num_heads, visualize_decoder_attn_weights=False,
            quantize_latent=True, factorize_latent=True, vq_codebook_size=4096, vq_token_dim=12, vq_commitment_cost=0.25, vq_use_l2_norm = True,
            num_init_latent_tokens=32, img_size=256, patch_size=16, max_rollout_iters=8,
            dynamic_halting=True, dynamic_halting_threshold=0.025, 
            train_stage="latent_distillation_pretrain"
        ):
        
        super().__init__()
        
        self.train_stage = train_stage
        self.quantize_latent = quantize_latent
        if quantize_latent is True: factorize_latent=True
        self.factorize_latent = factorize_latent
        self.dynamic_halting = dynamic_halting
        self.dynamic_halting_threshold = dynamic_halting_threshold
        self.max_rollout_iters = max_rollout_iters
        grid_size = img_size // patch_size
        scale = encoder_width ** -0.5

        self.encoder_ln_pre = nn.LayerNorm(encoder_width)
        self.encoder_ln_post = nn.LayerNorm(encoder_width)
        self.encoder_ln_recursive = nn.LayerNorm(encoder_width)
        self.pre_quantizer_mlp = nn.Linear(encoder_width, vq_token_dim, bias=True)
        self.encoder = Encoder(encoder_width, encoder_num_layers, encoder_num_heads)
        self.decoder = Decoder(decoder_width, decoder_num_layers, decoder_num_heads, factorize_latent=self.factorize_latent, factorized_latent_dim=vq_token_dim, output_dim=base_tokenizer.embed_dim, vis_attn_weights=visualize_decoder_attn_weights)

        self.encoder_positional_embedding = nn.Parameter(scale * torch.randn(grid_size ** 2 + 1, encoder_width))
        self.encoder_class_embedding = nn.Parameter(scale * torch.randn(1, encoder_width))
        self.encoder_mask_token = nn.Parameter(scale * torch.randn(1, 1, encoder_width))

        self.decoder_positional_embedding = nn.Parameter(scale * torch.randn(grid_size ** 2 + 1, decoder_width))
        self.decoder_class_embedding = nn.Parameter(scale * torch.randn(1, decoder_width))
        self.decoder_mask_token  = nn.Parameter(scale * torch.randn(1, 1, decoder_width))
        
        self.latent_tokens = nn.Parameter(scale * torch.randn(num_init_latent_tokens, encoder_width))
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(num_init_latent_tokens, encoder_width))
        self.timestep_embedding = nn.Parameter(scale * torch.randn(self.max_rollout_iters, num_init_latent_tokens, encoder_width))
        
        # num param: 775152 = (16*16)*3*1008 + 1008 [(kernal_size^2)*in_c*out_c + out_c (bias)]
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=encoder_width-base_tokenizer.embed_dim, # 1024 - 16 = 1008
            kernel_size=patch_size, stride=patch_size, bias=True)

        self.apply(self._init_weights)
        
        if self.quantize_latent:
            from modules.vector_quantizer import VectorQuantizer
            # Intialization for Quantizer is done inside VectorQuantizer
            self.quantize = VectorQuantizer(
                codebook_size=vq_codebook_size,
                token_size=vq_token_dim,
                commitment_cost=vq_commitment_cost,
                use_l2_norm=vq_use_l2_norm)
        
        self.base_tokenizer = base_tokenizer

        # TODO: Different loss weights per iteration might not be very critical
        self.lambda_loss_weight = [2.5, 2.0, 1.5, 1.25, 1.0, 1.0, 1.0, 1.0]
        
        if self.train_stage=="full_finetuning":
            # TODO: Ablate the requirement of different discriminators for different recurrent rollout iterations.
            # Intuition is at different rollout iteration .....
            from modules.losses.vqperceptual import VQLPIPSWithDiscriminator
            self.gan_losses = nn.ModuleList([VQLPIPSWithDiscriminator(
                disc_conditional= False, disc_in_channels= 3, 
                disc_start= 0, disc_weight= 0.2, codebook_weight= 1.0, # perceptual_weight=0.0
            ) for _ in range(self.max_rollout_iters)])
        
        self.visualize_decoder_attn_weights = visualize_decoder_attn_weights
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def preprocess_encoder(self, img_tokens):
        x = img_tokens
        x = torch.cat([_expand_token(self.encoder_class_embedding.to(x.get_device()), x.shape[0]), x], dim=1)
        x = x + self.encoder_positional_embedding
            
        latent_tokens = self.latent_tokens + self.latent_token_positional_embedding
        latent_tokens = latent_tokens[None].repeat(x.shape[0], 1, 1)
        return x, latent_tokens
    
    def preprocess_decoder(self, img_tokens):
        mask_tokens = self.decoder_mask_token.repeat(img_tokens.shape[0], img_tokens.shape[1], 1).to(img_tokens.dtype)
        mask_tokens = torch.cat([_expand_token(self.decoder_class_embedding, mask_tokens.shape[0]).to(mask_tokens.get_device()), mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.decoder_positional_embedding.to(mask_tokens.dtype)
        return mask_tokens
    
    def perform_dynamic_halting(self, x, decoded_code, gt_code, num_img_tokens):
        # To save compute, we skip performing dynamic halting at the last iteration, since its a waste.
        if iter!=self.max_rollout_iters-1: 
            drop_mask = torch.zeros_like(decoded_code[:,:,0])
            drop_mask[torch.abs(decoded_code-gt_code).mean(dim=-1)<self.dynamic_halting_threshold][:] = 1.

            img_tokens_minus_class = x[:,1:num_img_tokens]
            encoder_mask_token = self.encoder_positional_embedding[None,1:].repeat(x.shape[0],1,1) + self.encoder_mask_token.repeat(x.shape[0], img_tokens_minus_class.shape[1], 1)
            img_tokens_minus_class = encoder_mask_token * drop_mask[...,None] + (1-drop_mask[...,None]) * img_tokens_minus_class
            x = torch.cat((x[:,:1], img_tokens_minus_class, x[:,num_img_tokens:]), dim=1)
        return x
        
    def reconstruct_images(self, decoded_code):
        decoded_code = decoded_code.reshape(decoded_code.shape[0], 16, 16, decoded_code.shape[-1]).permute([0,3,1,2])
        return self.base_tokenizer.vae.decode(decoded_code)

    def encode(self, imgs, 
            return_min_length_embedding=True, 
            token_selection_criteria="reconstruction_loss", threshold=0.07, 
            return_embedding_type="latent_tokens"):
        
        # parameter return_all_embeddings returns multiple representations per image.
        # parameter return_min_length_embedding returns smallest length embedding with satisfies an objective (reconstruction loss < threshold for now).
        # parameter return_embedding_type \in ["latent_tokens", "image_and_latent_all_tokens", "image_tokens"], default="latent_tokens"
        
        # token selection criteria decides the satisfyable length of the embedding.
        # right now we only support reconstruction_loss as the automatic token selection criteria.
        # alternative TSC used in the paper require oracle / GT depth or class labels.
        # one could also learn a token selection criteria based on input image (we might release this at some point)

        reconstruction_iters = []
        if return_min_length_embedding: 
            assert(return_embedding_type=="latent_tokens")
            best_tsc, best_tsc_iter = torch.inf, -1 # tsc = token selection criteria                 
        
        all_logs = self.forward(imgs, return_image_embeddings=True, reconstruction_iters="all")
        all_embeddings = []
        all_reconstructions = []
        for iter, iter_logs_dict in enumerate(all_logs):
            for key in iter_logs_dict.keys():
                if return_embedding_type in key:
                    all_embeddings.append(iter_logs_dict[key])
                    all_reconstructions.append(iter_logs_dict["reconstructed_imgs_{}".format(key.split("_")[-1])])
                    if return_min_length_embedding:
                        if token_selection_criteria=="reconstruction_loss":
                            reconstructed_imgs = iter_logs_dict["reconstructed_imgs_{}".format(key.split("_")[-1])]
                            loglaplace_loss = torch.abs(reconstructed_imgs - imgs).mean()
                            if loglaplace_loss < best_tsc:
                                best_tsc = loglaplace_loss
                                best_tsc_embed = iter_logs_dict[key]
                                best_tsc_reconstruction = reconstructed_imgs
                                best_tsc_iter = iter
                            if best_tsc < threshold:
                                # if already < threshold return the embedding and corresponding reconstruction.
                                return best_tsc_embed, best_tsc_reconstruction
        
        # if threshold cannot be satisfied, return max tokens
        if return_min_length_embedding:
            return best_tsc_embed, best_tsc_reconstruction

        return all_embeddings, all_reconstructions, all_logs

    
    def forward(self, imgs, sample_grad_iters=-1, reconstruction_iters=[], gan_optimizer_idx=None, gan_loss_weight=None, return_image_embeddings=False):
        # sample_grad_iters==-1: evaluate loss at all roll out iterations (default setting). 
        # Otherwise, we randomly evaluate loss at sample_grad_iters number of iterations.
        # reconstruction_iters==[] – reconstruct back images at different iterations (default setting).
        # reconstruction_iters=="grad" – reconstruct back images at all gradient iters.
        # reconstruction_iters=="all" – reconstruct back images at all iters

        # Generating image tokens and pre-trained vae tokens.
        # Initializing masked_2D_tokens and init_latent_tokens
        
        # imgs: [2, 3, 256, 256]
        vae_tokens = self.base_tokenizer.get_img_tokens(imgs) # [2, 16, 16, 16]
        img_tokens = self.patch_embed(imgs) # [1, 1008, 16, 16]
        img_tokens = torch.cat((vae_tokens, img_tokens), dim=1) # [2, 1024, 16, 16]
        vae_tokens = vae_tokens.reshape(vae_tokens.shape[0], vae_tokens.shape[1], -1).permute([0,2,1]) # [2, 256, 16] [bsz, num_latent_pixel, vae_latent_channel]
        img_tokens = img_tokens.reshape(img_tokens.shape[0], img_tokens.shape[1], -1).permute([0,2,1]) # [2, 256, 1024] [bsz, num_latent_pixel, img_token_dim]
        img_tokens = F.normalize(img_tokens, dim=-1) # [2, 256, 1024]
        gt_code = vae_tokens # [2, 256, 16]
        
        # class token is always concat at before the image tokens.

        # decoder_mask_token = nn.Parameter([1, 1, embed_dim=1024]) <= per latent pixel
        # decoder_class_embedding = nn.Parameter([1, 1024]) <= per image
        # then, expand the decoder_mask_token to [2, 256, 1024] and decoder_class_embedding to [2, 1, 1024]
        # then, concat the two to get [2, 257, 1024]
        # then for each batch add learnable positional embedding [257, 1024] => final output [2, 257, 1024]
        # IMPORTANT: the masked_2d_tokens initially does not have any information from the image tokens.
        masked_2d_tokens = self.preprocess_decoder(img_tokens) # [2, 257, 1024]
        
        # encoder_class_embedding = nn.Parameter([1, 1024]) <= per image
        # encoder_positional_embedding = nn.Parameter([257, 1024]) <= per image
        # img_token is concated with the encoder_class_embedding and added with positional embedding
        # latent tokens is initialized with nn.Parameter([32, 1024]) and added with positional embedding nn.Parameter([32, 1024])
        img_tokens, init_latent_tokens = self.preprocess_encoder(img_tokens) # [2, 257, 1024], [2, 32, 1024]
        num_img_tokens = img_tokens.shape[1]
        
        # self.timestep_embedding = nn.Parameter(self.max_rollout_iters, 32=num_init_latent_token, 1024)
        # here, going to start the first iter, so use the first timestep embedding
        x = torch.cat([img_tokens, init_latent_tokens + self.timestep_embedding[0]], dim=1) # [2, 289, 1024]
        x = self.encoder_ln_pre(x) # LayerNorm, [2, 289, 1024]
        
        # Sampling rollout iterations at which gradient should be computed.
        if isinstance(sample_grad_iters, list):
            # In full_finetuning stage we compute loss at only one iteration which comes from engines/full_finetuning.py
            grad_iters = sample_grad_iters
        else:
            grad_iters = np.arange(self.max_rollout_iters)
            if self.training and sample_grad_iters!=-1:
                np.random.shuffle(grad_iters)
                grad_iters = grad_iters[:sample_grad_iters] # randomly sample the stage to get grad
            grad_iters = grad_iters.tolist()
            
        # the recons loss is computed at reconstruction_iters stage
        if reconstruction_iters=="grad": reconstruction_iters=grad_iters
        elif reconstruction_iters=="all": reconstruction_iters = np.arange(self.max_rollout_iters).tolist()

        all_logs = []
        total_loss = 0
        for iter in range(self.max_rollout_iters):
            # image_tokens, initialized_latent_tokens -> processed image_tokens, learned latent_tokens
            
            # multihead self attention to merge info between image tokens and latent tokens
            # at each iter, x is [class_token, GT_image_tokens_*without*_tokens_that_are_well_reconstructed_now, latent_tokens]
            # here :
            #   - GT_image_tokens_*without*_tokens_that_are_well_reconstructed_now is the information source to encode
            #   - latent_tokens is the information source to decode
            # technically, although here the author uses self-attn, it should be cross-attn where latent gather information from image tokens
            x = self.encoder(x) # [2, 257 + 32*iter, 1024] -> [2, 257 + 32*iter, 1024]
            latent_tokens = x[:, img_tokens.shape[1]:] # [2, 32*iter, 1024]
            
            # Latent quantization and decoding is only required either for image reconstruction at test time or for computing reconstruction loss at train time.  
            # To save compute at train time, one could randomly sample different iterations at which gradient should be computed.
            if not self.training or iter in grad_iters or iter in reconstruction_iters:
                iter_logs_dict = {}
                if return_image_embeddings:
                    iter_logs_dict.update({
                        "image_and_latent_all_tokens_{}".format(iter): x[:,1:], # remember, the first token is the class token
                        "image_tokens_{}".format(iter): x[:,1:num_img_tokens], # ignoring the class token, class token had no form of learning signal during training.
                        "latent_tokens_{}".format(iter): latent_tokens
                    })
                latent_tokens = self.encoder_ln_post(latent_tokens) # LayerNorm, [2, 32*iter, 1024]
                
                if self.factorize_latent: latent_tokens_factorized = self.pre_quantizer_mlp(latent_tokens) # just a down-proj MLP: [2, 32*iter, 12]
                else: latent_tokens_factorized = latent_tokens # No factorization performed.
                
                if self.quantize_latent:
                    latent_tokens_quantized, quant_result_dict = self.quantize(latent_tokens_factorized, is_quantize=True)
                    if self.visualize_decoder_attn_weights:
                        decoded_code, decoded_attn_weights = self.decoder(latent_tokens_quantized, masked_2d_tokens)
                        iter_logs_dict.update({"decoded_attn_weights_{}".format(iter): decoded_attn_weights})
                    else:
                        decoded_code = self.decoder(latent_tokens_quantized, masked_2d_tokens)
                else:
                    if self.visualize_decoder_attn_weights:
                        decoded_code, decoded_attn_weights = self.decoder(latent_tokens_factorized, masked_2d_tokens)
                        iter_logs_dict.update({"decoded_attn_weights_{}".format(iter): decoded_attn_weights})
                    else:
                        # cross attention
                        # masked_2d_tokens get information from latent tokens
                        # since previously, latent_tokens **further** gathered information from image tokens
                        # so, now, masked_2d_tokens **further** get information from image tokens, to form the decoded_code
                        decoded_code = self.decoder(latent_tokens_factorized, masked_2d_tokens) # [2, 32*iter, 12],[2, 257, 1024] -> [2, 256, 16]
                
                if self.dynamic_halting:
                    # the purpose is to mask out good enough image tokens
                    # remember, x =[class_token, GT_image_tokens, latent_tokens]
                    # in perform_dynamic_halting, we only update the image tokens
                    # the process is:
                    #   - compare the decoded_code and gt_code (both are [2, 256, 16])
                    #   - if for a latent pixel, the difference is small, mask out (replace) the image token with `self.encoder_mask_token`
                    # Therefore, now, the x becomes: [class_token, GT_image_tokens_*without*_tokens_that_are_well_reconstructed_now, latent_tokens]
                    x = self.perform_dynamic_halting(x, decoded_code, gt_code, num_img_tokens) # [2, 289, 1024] -> [2, 289, 1024]
                
                if self.training and iter in grad_iters:
                    if self.train_stage == "latent_distillation_pretrain":
                        iter_code_loss = self.forward_loss(gt_code, decoded_code)
                        total_loss = total_loss + self.lambda_loss_weight[iter] * iter_code_loss
                        iter_logs_dict.update({
                            "code_loss_{}".format(iter): iter_code_loss.item()
                        })
                    elif self.train_stage == "full_finetuning":
                        reconstructed_imgs = self.reconstruct_images(decoded_code)
                        total_loss, iter_logs_dict = self.forward_gan_losses(imgs, reconstructed_imgs, gan_optimizer_idx, iter_idx=iter, discriminator_loss_weight=gan_loss_weight)
                        iter_logs_dict.update({
                            "reconstructed_imgs_{}".format(iter): reconstructed_imgs,
                        })

                    if self.quantize_latent: 
                        total_loss = total_loss + 1. * quant_result_dict['quantizer_loss']
                        iter_logs_dict.update({
                            "quantization_loss_{}".format(iter): quant_result_dict['quantizer_loss'].item(),
                        })

                if iter in reconstruction_iters and "reconstructed_imgs_{}".format(iter) not in iter_logs_dict:
                    reconstructed_imgs = self.reconstruct_images(decoded_code)
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}".format(iter): reconstructed_imgs,
                    })  

                all_logs.append(iter_logs_dict)

            # TODO Ablation -- timestep_embedding is not a critical component, can be avoided. Would enable infinite rollout at test time.
            if iter!=self.max_rollout_iters-1:
                # now add in more latent tokens to get information
                x = torch.cat((x, init_latent_tokens + self.timestep_embedding[iter+1]), dim=1)
                x = self.encoder_ln_recursive(x)
        
        if not self.training: return all_logs
        return total_loss, all_logs
    
    def forward_loss(self, gt_code, decoded_code):
        code_loss = (gt_code - decoded_code)**2
        return code_loss.mean()

    def get_last_layer(self):
        return self.base_tokenizer.vae.decoder.conv_out.weight

    def forward_gan_losses(self, imgs, reconstructed_imgs, optimizer_idx, iter_idx, discriminator_loss_weight):
        assert(optimizer_idx is not None)
        if discriminator_loss_weight==0:
            global_step=-torch.inf
            self.gan_losses[iter_idx].discriminator_weight = 0.2
        else:
            global_step=torch.inf
            self.gan_losses[iter_idx].discriminator_weight = discriminator_loss_weight
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")

            iter_log_dict_ae = {}
            for key in log_dict_ae.keys():
                iter_log_dict_ae["{}_{}".format(key, iter_idx)] = log_dict_ae[key]
            return aeloss, iter_log_dict_ae

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")
            
            iter_log_dict_disc = {}
            for key in log_dict_disc.keys():
                iter_log_dict_disc["{}_{}".format(key, iter_idx)] = log_dict_disc[key]
            
            return discloss, iter_log_dict_disc


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torch
    import adaptive_tokenizers
    from utils import misc

    # set arguments accordingly
    args = {
        'image_path': '/data2/czhenghao/adaptive-length-tokenizer/assets/custom_images/birds/000001.png',
        'device': 'cuda:0',
        'input_size': 256,
        'model': 'alit_small',
        'base_tokenizer': 'vae',
        # 'ckpt': 'adaptive_tokenizers/pretrained_models/imagenet100/alit_small_vae_continuous_latents.pth',
        'quantize_latent': False
    }
    args = misc.Args(**args)


    image = Image.open(args.image_path).convert("RGB")
    transform_val = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor()
    ])
    image_tensor = transform_val(image).to(args.device)[None]
    image_tensor = image_tensor.repeat(2,1,1,1)

    plt.imshow(image_tensor[0].permute([1,2,0]).cpu().numpy())
    plt.savefig("input_image.png")

    base_tokenizer_args = {
        "id": args.base_tokenizer,
        "is_requires_grad": False
    }
    adaptive_tokenizer = adaptive_tokenizers.__dict__[args.model](
        base_tokenizer_args=base_tokenizer_args, quantize_latent=args.quantize_latent, 
        train_stage="full_finetuning")

    adaptive_tokenizer.to(args.device)
    adaptive_tokenizer.eval()

    all_embeddings, all_reconstructions, all_logs = adaptive_tokenizer.encode(image_tensor, return_min_length_embedding=False, token_selection_criteria="reconstruction_loss", threshold=0.05, return_embedding_type="latent_tokens")
