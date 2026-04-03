clear all
close all
clc

% Author: Stefano Fortunati (2026)

% Monte carlo runs
Ns=10^5;

% "Small perturbation" term for the R-estimator
perturbation_par = 0.01;

% nu parameter for the t-based score function
nu_par = 4;

% Maximum number of iterations for the Tyler's estimator
Max_it_Ty = 10;

% Maximum number of iterations for the R-estimator
Max_it_R = 3;

% Dimension of the observation vector
m = 8;

% Number of observations
n = 100;

% Scatter matrix
ro=0.8;
rx=ro.^[0:m-1];
Sigma = toeplitz(rx);
Inv_Sigma = inv(Sigma);
L=chol(Sigma);
L=L';

% Degree of freedom of the t-distribution
nuvect=[2.1 3:1:20];
Nl=length(nuvect);

% Duplication, elimination and commutation matrices
D_m = full(DuplicationM(m));
K_m = CommutationM(Sigma);


DIM = m*(m+1)/2;

I_bar = eye(DIM);
I_bar(1,:) = [];
I_DIM_1 = eye(DIM-1);
I_m2 = eye(m^2);
I_m = eye(m);
vec_I_m = I_m(:);
D_m_diesis = pinv(D_m);

%%%%%% For S(Sigma) = det(Sigma)^(1/m)
V_S = Sigma/(det(Sigma)^(1/m));
vec_V_S = V_S(:);
Inv_V_S = inv(V_S);
P_S = I_m2-(1/m)*V_S(:)*Inv_V_S(:).';
vec_Inv_V_S = Inv_V_S(:);
T_1_appo = vec_Inv_V_S.'*D_m;
grad_K = -(T_1_appo*I_bar.')/T_1_appo(1,1);
K_V_S = [grad_K ; I_DIM_1];
M_V_S = (D_m*K_V_S).';
Kprod_Vs = kron(V_S,V_S);
Invrad_Vs = inv(sqrtm(V_S));
Kprod_Inv_sqrtm_Vs = kron(Invrad_Vs,Invrad_Vs);

tic
for il=1:Nl

    nu = nuvect(il)

    MSE_SCM = 0;
    MSE_Ty = 0;
    MSE_R_vdW = 0;
    MSE_R_t_nu = 0;

    parfor ins=1:Ns

        w = randn(m,n);
        R = gamrnd(nu/2,2/nu,1,n);
        x = L*w;
        y = sqrt(1./(repmat(R,m,1))).*x;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sample Covariance Matrix
        SCM = y*y'/n;

        %%%%%% For S(Sigma) = det(Sigma)^(1/m)
        V_SCM = SCM/(det(SCM)^(1/m));

        err_SCM = V_SCM(:)-vec_V_S;
        MSE_SCM = MSE_SCM + (err_SCM.'*err_SCM)/Ns;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tyler's estimator
        Ty = Tyler_R_11(y, Max_it_Ty);

        %%%%%% For S(Sigma) = det(Sigma)^(1/m)
        V_Ty = Ty/(det(Ty)^(1/m));

        err_Ty = V_Ty(:)-vec_V_S;
        MSE_Ty= MSE_Ty + (err_Ty.'*err_Ty)/Ns;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constrained R-estimator with the van der Waerden score function
        [V_RvdW, ~] = R_vdW_est_const_det_iter(y, V_Ty, perturbation_par, Max_it_R);

        err_RvdW = V_RvdW(:)-vec_V_S;
        MSE_R_vdW = MSE_R_vdW + (err_RvdW.'*err_RvdW)/Ns;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constrained R-estimator with the t_based score function
        [V_Rt, ~] = R_t_nu_est_const_det_iter(y, V_Ty, nu_par, perturbation_par, Max_it_R);

        err_Rt = V_Rt(:)-vec_V_S;
        MSE_R_t_nu = MSE_R_t_nu + (err_Rt.'*err_Rt)/Ns;

    end
    Fro_MSE_SCM(il) = MSE_SCM;
    Fro_MSE_Ty(il) = MSE_Ty;
    Fro_MSE_R_vdW(il) = MSE_R_vdW;
    Fro_MSE_R_t_nu(il) = MSE_R_t_nu;

    alpha_0 = (nu + m)/(m+2+nu);

    % SCRB on underbar_vecs V
    SCRB = alpha_0^(-1)*I_bar*D_m_diesis*P_S*(eye(m^2)+K_m)*Kprod_Vs*P_S.'*D_m_diesis.'*I_bar.'/n;
    FULL_SCRB = M_V_S.'* SCRB * M_V_S;

    % CRB on vecs V
    I_ub_V = 1/4* M_V_S * Kprod_Inv_sqrtm_Vs *(alpha_0*(eye(m^2)+K_m) + (alpha_0-1)*(vec_I_m*vec_I_m.')) * Kprod_Inv_sqrtm_Vs * M_V_S.';
    CRB_V = inv(I_ub_V)/n;
    FULL_CRB = M_V_S.'* CRB_V * M_V_S;

    SCR_Bound(il) = trace(FULL_SCRB);
    CR_Bound(il) = trace(FULL_CRB);

end


color_matrix(1,:)=[0 0 1]; % Blue
color_matrix(2,:)=[1 0 0]; % Red
color_matrix(3,:)=[0 0.5 0]; % Dark Green
color_matrix(4,:)=[0 0 0]; % Black
color_matrix(5,:)=[0 0.5 1]; % Light Blue
color_matrix(6,:)=[1 0.3 0.6]; % Pink
color_matrix(7,:)=[0 0.9 0]; % Light Green

line_marker{1}='-s';
line_marker{2}='--d';
line_marker{3}=':^';
line_marker{4}='-.p';
line_marker{5}='-o';
line_marker{6}='--h';
line_marker{7}='-.*';

figure(1)
semilogy(nuvect,CR_Bound,line_marker{1},'LineWidth',1,'Color',color_matrix(1,:),'MarkerEdgeColor',color_matrix(1,:),'MarkerFaceColor',color_matrix(1,:),'MarkerSize',8);
hold on
semilogy(nuvect,SCR_Bound,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:),'MarkerSize',8);
hold on
semilogy(nuvect,Fro_MSE_SCM,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:),'MarkerSize',8);
hold on
semilogy(nuvect,Fro_MSE_Ty,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
hold on
semilogy(nuvect,Fro_MSE_R_vdW,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
hold on
semilogy(nuvect,Fro_MSE_R_t_nu,line_marker{6},'LineWidth',1,'Color',color_matrix(6,:),'MarkerEdgeColor',color_matrix(6,:),'MarkerFaceColor',color_matrix(6,:),'MarkerSize',8);
grid on
axis([2.1 20 0 2*max(Fro_MSE_Ty)])
xlabel('Degrees of freedom: $\nu$','interpreter','latex');ylabel('MSE and bounds','interpreter','latex');
legend('CRB on $\mathrm{vec}(\mathbf{V}_S)$','SCRB on $\mathrm{vec}(\mathbf{V}_S)$','Const. SCM','Const. Tyler','U-const. $R_{vdW}$','U-const. $R_{t_{\nu}$','interpreter','latex')
title('Estimation of $\mathrm{vec}(\mathbf{V}_S)$ with $S(\mathbf{\Sigma})=|\mathbf{\Sigma}|^{1/m}$','interpreter','latex')
