function [Delta_T_1, KK_1] = Delta_Psi_eval_vdW_const_11(y, T, D_n)

% Author: Stefano Fortunati (2026)

[m, n] = size(y);
T = T/T(1,1);

% Evaluation of the score function and of the vector u
[score_vect,u,inv_sr_T] = score_rank_sign_vdW(y,T);

% Evaluation of the matrix K_V for the contraint 11
inv_sr_T2 = kron(inv_sr_T.',inv_sr_T);
I_m = eye(m);
J_m_per = eye(m^2) - I_m(:)*I_m(:).'/m;
M_m = D_n(:,2:end).';
K_V_1 = M_m*(inv_sr_T2*J_m_per);

%%%% Evaluation of the approximation of the efficient central sequence
%%% Pedagogical version of the calculation
% Score_appo = zeros(N^2,1);
% for k=1:K
%    Mat_appo = (u(:,k)*u(:,k)');
%    Score_appo = Score_appo  + score_vect(k)*Mat_appo(:);
% end

%%% Fast version of the calculation
Mat_appo = u .* reshape( u', [1 n m] );
Mat_appo = reshape( permute( Mat_appo, [1 3 2] ), m^2, [] );
Score_appo = Mat_appo*score_vect.';

Delta_T_1 = (1/2)*K_V_1*Score_appo/sqrt(n);
KK_1 = (1/2)*(K_V_1*K_V_1.');
end
