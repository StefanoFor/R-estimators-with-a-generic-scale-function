function alpha_est = alpha_estimator_sub_t_nu_const(y, T, nu, pert, D_m, E_m)

% Author: Stefano Fortunati (2026)

[m, n] = size(y);

% Definition of the "small perturbation" matrix 
V = pert*randn(m,m);
V = (V+V')/2;
V(1,1)=0;

% Evaluation of the approximation of the efficient central sequence Delta_T for the constraint 11
% and of the matrix Psi_T, where T is the preliminary estimator
[Delta_T_1, KK_1] = Delta_Psi_eval_t_nu_const_11(y, T, nu, D_m);

% Evaluation of the perturbed approximation of the efficient central sequence Delta_T
T = T/T(1,1);
T_pert = T + V/sqrt(n);

Delta_T_1_pert = Delta_only_eval_t_nu_const_11(y, T_pert, nu, D_m);

% Estimation of alpha
V_1 = V(:);
N_m = E_m(2:end,:);
alpha_est = norm(Delta_T_1_pert-Delta_T_1)/norm(KK_1*(N_m*V_1));
end

