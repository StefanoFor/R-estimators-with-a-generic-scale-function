function [ V_VDW,  alpha_est] = R_t_nu_est_const_trace( y, T, nu, pert)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: 
% Real-valued data matrix: y (The data is assumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary consistent estimator: T 
% Parameter of the t-based score function: nu
% Perturabtion parameter: pert

% Output:
% Semiparametric efficient R-estimator of the shape: V_VCW
% Convergence parameter: beta_est

% The score function used here is the t_nu-based score.

% Author: Stefano Fortunati (2026)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(y);

% Calculation of the Constraint Matrix U_V for the constraint S(Sigma) = trace(Sigma)/m
I_m = eye(m);
D_Sigma = m^(-1)*I_m;
U_V = null(D_Sigma(:)');



% Estimation of alpha
alpha_est= alpha_estimator_sub_t_nu_const( y, T, nu, pert);


[Delta_T, inv_Psi_T] = Delta_Psi_eval_t_nu_const(y, T, nu, U_V);

% Vectorized one-step estimatimator of the shape matrix
N_VDW_vec = T(:) + (inv_Psi_T*Delta_T)/(alpha_est*sqrt(n));

% One-step estimatimator of the shape matrix
V_VDW = reshape(N_VDW_vec, [m,m]);
V_VDW = (V_VDW+V_VDW')/2;


end
