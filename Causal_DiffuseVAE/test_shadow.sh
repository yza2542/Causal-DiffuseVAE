python inference_cvae_shadow.py +dataset=shadow/test \
                      dataset.vae.data.root='/home/2527823Y/DiffuseVAE-main/main/datasets/Shadow_Do' \
                      dataset.vae.data.name='mnist' \
                      dataset.vae.evaluation.batch_size=64 \
                      dataset.vae.evaluation.dataset_dir='/home/2527823Y/DiffuseVAE-main/main/datasets/Shadow_Do' \
                      dataset.vae.evaluation.epochs=100 \
                      dataset.vae.evaluation.device=\'gpu:0\' \
                      dataset.vae.evaluation.results_dir=\'/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/checkpoint_shadow\' \
                      dataset.vae.evaluation.workers=16 \
                      dataset.vae.evaluation.chkpt_prefix=\'celeba64_alpha=1.0\' \
                      dataset.vae.evaluation.alpha=1.0 \
                      dataset.vae.evaluation.chkpt_path=\'/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/checkpoint_shadow_Do/checkpoints/vae_celeba_128-celeba64_alpha=1.0-epoch=99-train_loss=0.0000.ckpt\'
                      #dataset.vae.evaluation.chkpt_path=\'/home/2527823Y/causal_save/result_mse_weightkl_0.1/checkpoints/vae_celeba_128-celeba64_alpha=1.0-epoch=99-train_loss=0.0000.ckpt\'
                      #dataset.vae.evaluation.chkpt_path=\'/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/checkpoint/checkpoints/vae_celeba_128-celeba64_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\'
