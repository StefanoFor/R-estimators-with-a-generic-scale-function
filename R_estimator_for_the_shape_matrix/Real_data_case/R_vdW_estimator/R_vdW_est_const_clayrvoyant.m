function V_VDW = R_vdW_est_const_clayrvoyant( y, T, alpha_0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: 
% Real-valued data matrix: y (The data is assumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary consistent estimator: T 
% Matrix related to the constraints: U
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

% Calculation of the Constraint Matrix U_V
D_Sigma = m^(-1)*(det(T)^(1/m))*inv(T);
U_V = null((E_m*D_Sigma(:)).');

[Delta_T, inv_Psi_T] = Delta_Psi_eval_vdW_const(y, T, D_m, U_V);

% Vectorized one-step estimatimator of the shape matrix
N_VDW_vec = E_m*T(:) + (inv_Psi_T*Delta_T)/(alpha_0*sqrt(n));

% One-step estimatimator of the shape matrix
V_VDW = reshape(D_m*N_VDW_vec, [m,m]);


end
