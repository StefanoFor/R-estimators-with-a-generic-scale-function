function [ V_VDW,  alpha_est] = R_vdW_est_const_11( y, T, pert)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: 
% Real-valued data matrix: y (The data is assumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary consistent estimator: T 
% Perturabtion parameter: pert

% Output:
% Semiparametric efficient R-estimator of the shape: V_VCW
% Non-parametric estimation of alpha: alpha_est

% The score function used here is the van der Waerden score.

% Author: Stefano Fortunati (2026)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(y);

% Definition of the Duplication and Elimination matrices
D_m = full(DuplicationM(m));
E_m = full(EliminationM(m));

% Calculation of the Constraint Matrix U_V for the constraint S(Sigma) = Sigma(1,1)
e_1m = [1; zeros(m-1,1)];
D_Sigma = e_1m*e_1m.';
U_V = null((E_m*D_Sigma(:)).');


% Estimation of alpha
alpha_est= alpha_estimator_sub_vdW_const( y, T, pert, D_m, E_m);


[Delta_T, inv_Psi_T] = Delta_Psi_eval_vdW_const(y, T, D_m, U_V);

% Vectorized one-step estimatimator of the shape matrix
N_VDW_vec = E_m*T(:) + (inv_Psi_T*Delta_T)/(alpha_est*sqrt(n));

% One-step estimatimator of the shape matrix
V_VDW = reshape(D_m*N_VDW_vec, [m,m]);
V_VDW = (V_VDW+V_VDW')/2;

end
