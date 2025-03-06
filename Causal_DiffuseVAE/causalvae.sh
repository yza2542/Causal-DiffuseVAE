python train_cvae.py +dataset=shadow/train \
                      dataset.vae.data.root='/home/2527823Y/DiffuseVAE-main/main/datasets/flow_noise' \
                      dataset.vae.data.name='flow' \
                      dataset.vae.training.batch_size=64 \
                      dataset.vae.training.dataset_dir='/home/2527823Y/DiffuseVAE-main/main/datasets/flow_noise' \
                      dataset.vae.training.epochs=500 \
                      dataset.vae.training.device=\'gpu:0\' \
                      dataset.vae.training.results_dir=\'/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/checkpoint_flow\' \
                      dataset.vae.training.workers=16 \
                      dataset.vae.training.chkpt_prefix=\'celeba64_alpha=1.0\' \
                      dataset.vae.training.alpha=1.0
                      # dataset.vae.evaluation.chkpt_path=\'/home/2527823Y/causal_save/result_mse_weightkl_0.1/checkpoints/vae_celeba_128-celeba64_alpha=1.0-epoch=99-train_loss=0.0000.ckpt\'

# python train_cvae.py +dataset=Age/train \
#                      dataset.vae.data.root='/home/2527823Y/DiffuseVAE-main/main/datasets/Age_sub_new' \
#                      dataset.vae.data.name='age' \
#                      dataset.vae.training.batch_size=64 \
#                      dataset.vae.training.dataset_dir='/home/2527823Y/DiffuseVAE-main/main/datasets/Age_sub_new' \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0\' \
#                      dataset.vae.training.results_dir=\'/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/checkpoint_age\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix=\'celeba64_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

#python inference_cvae.py +dataset=smile_A/test \
#                      dataset.vae.data.root='/home/2527823Y/DiffuseVAE-main/main/datasets/smile64' \
#                      dataset.vae.data.name='smile' \
#                      dataset.vae.evaluation.batch_size=256 \
#                      dataset.vae.evaluation.dataset_dir='/home/2527823Y/DiffuseVAE-main/main/datasets/smile64' \
#                      dataset.vae.evaluation.epochs=100 \
#                      dataset.vae.evaluation.device=\'gpu:0\' \
#                      dataset.vae.evaluation.results_dir=\'/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/checkpoints\' \
#                      dataset.vae.evaluation.workers=4 \
#                      dataset.vae.evaluation.chkpt_prefix=\'celeba64_alpha=1.0\' \
#                      dataset.vae.evaluation.alpha=1.0 \
#                      dataset.vae.evaluation.chkpt_path=\'/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/checkpoints_lastversion/vae-celeba64_alpha=1.0-epoch=299-train_loss=0.0000.ckpt\'