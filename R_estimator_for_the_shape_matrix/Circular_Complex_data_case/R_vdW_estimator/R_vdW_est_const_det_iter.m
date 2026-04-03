function [ V_VDW,  alpha_est] = R_vdW_est_const_det_iter( y, T, pert, iter)

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

% Estimation of alpha
alpha_est= alpha_estimator_sub_vdW_const( y, T, pert);

for i = 1:iter
    % Calculation of the Constraint Matrix U_V for the constraint S(Sigma) = det(Sigma)^(1/m)
    D_Sigma = m^(-1)*(det(T)^(1/m))*inv(T);
    U_V = null(D_Sigma(:)');

    [Delta_T, inv_Psi_T] = Delta_Psi_eval_vdW_const(y, T, U_V);

    % Vectorized one-step estimatimator of the shape matrix
    N_VDW_vec = T(:) + (inv_Psi_T*Delta_T)/(alpha_est*sqrt(n));

    % One-step estimatimator of the shape matrix
    V_VDW = reshape(N_VDW_vec, [m,m]);

    T =  V_VDW/(det(V_VDW)^(1/m));
end

end
