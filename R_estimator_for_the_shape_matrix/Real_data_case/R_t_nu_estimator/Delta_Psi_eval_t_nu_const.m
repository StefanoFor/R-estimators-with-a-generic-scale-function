function [Delta_T, inv_Psi_T] = Delta_Psi_eval_t_nu_const(y, T, nu, D_m, U_c)

% Author: Stefano Fortunati (2026)

[m, n] = size(y);

% Evaluation of the score function and of the vector u
[score_vect,u,inv_sr_T] = score_rank_sign_t_nu(y,T,nu);

% Evaluation of the "unconstrained" matrix K_V
inv_sr_T2 = kron(inv_sr_T.',inv_sr_T);
I_m = eye(m);
J_m_per = eye(m^2) - I_m(:)*I_m(:).'/m;
K_V = D_m.'*(inv_sr_T2*J_m_per);

%%%% Evaluation of the approximation of the efficient central sequence in Eq. (33) of [a] 
%%% Pedagogical version of the calculation
% Score_appo = zeros(m^2,1);
% for k=1:n
%    Mat_appo = (u(:,k)*u(:,k)');
%    Score_appo = Score_appo  + score_vect(k)*Mat_appo(:);
% end

%%% Fast version of the calculation
Mat_appo = u .* reshape( u', [1 n m] );
Mat_appo = reshape( permute( Mat_appo, [1 3 2] ), m^2, [] );
Score_appo = Mat_appo*score_vect.';

Delta_T = (1/2)*K_V*Score_appo/sqrt(n);
KK = (1/2)*(K_V*K_V.');

% Definition of the constrained matrix Psi 
inv_Psi_T = U_c*inv(U_c'*KK*U_c)*U_c';


end

