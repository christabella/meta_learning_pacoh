guild run 1D:train-np -y
guild run 1D:train-np -y meta_learner=conditional_neural_process
guild run 1D:train-np -y meta_learner=attentive_neural_process

guild run 1D:train-meta-gpr -y dataset=mnist mean_module=zero
guild run 1D:train-meta-gpr -y dataset=mnist mean_module=constant
guild run 1D:train-meta-gpr -y dataset=mnist mean_module=NN
guild run 1D:train-meta-gpr -y dataset=mnist mean_module=NN covar_module=SE
guild run 1D:train-meta-gpr -y dataset=mnist mean_module=NN covar_module=NN..................
