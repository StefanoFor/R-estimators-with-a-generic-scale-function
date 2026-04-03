function [score_vect,u,SR_IN_T] = score_rank_sign_t_nu(y,T,nu)

% Author: Stefano Fortunati (2026)

[m, n] = size(y);

% Evaluation of the square root of the inverse of the preliminary estimator T
SR_IN_T = inv(sqrtm(T));

% Evaluation of Q^\star
Q = dot(SR_IN_T*y,SR_IN_T*y);

% Evaluation of u^\star
u = SR_IN_T*y./sqrt(Q);

% Evaluation of the ranks r^\star of Q^\star
[~,p] = sort(Q,'ascend');
r = 1:n;
r(p) = r;

% Evaluation of the t_nu-score function
f_inv_appo = finv(r/(n+1),2*m,nu);
score_vect = m*f_inv_appo.*(2*m + nu)./(nu + 2*m*f_inv_appo);
end

