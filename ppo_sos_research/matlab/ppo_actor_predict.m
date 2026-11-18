function [action, u_rl] = ppo_actor_predict(actor, observation)
% actor = load('actor_model_free_mu1.0.mat'); observation is a column.
% Build history in contract_json order, newest first; zero all buffers at reset.
% The caller supplies y_k (and, for controller_lti, normalized theta_base).
% No normalization statistics or hidden controller states are omitted.
% u_rl is physical in model_free/residual_rl; LTI's second output is NOT u.
z = single(observation(:));
assert(numel(z) == size(actor.W1, 2), 'Incompatible observation dimension');
for i = 1:double(actor.n_layers)
    W = single(actor.(sprintf('W%d', i)));
    b = single(actor.(sprintf('b%d', i)));
    z = W*z + b;
    if i < double(actor.n_layers)
        z = tanh(z);
    end
end
action = min(max(z, single(actor.action_low(:))), single(actor.action_high(:)));
u_rl = single(actor.physical_action_scale)*action;
% Apply clip(u_rl,-force_limit,force_limit) in model_free.
% Apply clip(u_base+u_rl,-force_limit,force_limit) in residual_rl.
% After the physical step, push y_k and the APPLIED u_sat,k into the histories.
end
