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

% Dimension of the observation vector
m = 4;

% Number of observations
n = 100;

% Scatter matrix
ro=0.8*exp(1j*2*pi/5);
rx=ro.^[0:m-1];
Sigma = toeplitz(rx);
Inv_Sigma = inv(Sigma);
L=chol(Sigma);
L=L';

% Degree of freedom of the t-distribution
nuvect=[2.1 3:1:20];
Nl=length(nuvect);

%%%%%% For S(Sigma) = trace(Sigma)/m
V_S = m*Sigma/trace(Sigma);
Inv_V_S = inv(V_S);
theta_true = V_S(:);

% Calculation of the Constraint Matrix U_V for the constraint S(Sigma) = trace(Sigma)/m
I_m = eye(m);
D_Sigma = m^(-1)*I_m;
U = null(D_Sigma(:)');

tic
for il=1:Nl

    nu = nuvect(il)

    MSE_SCM = 0;
    MSE_Ty = 0;
    MSE_R_vdW = 0;
    MSE_R_t_nu = 0;

    parfor ins=1:Ns

        w = (randn(m,n) + 1j*randn(m,n))/sqrt(2);
        R = gamrnd(nu/2,2/nu,1,n);
        x = L*w;
        y = sqrt(1./(repmat(R,m,1))).*x;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sample Covariance Matrix
        SCM = y*y'/n;

        %%%%%% For S(Sigma) = trace(Sigma)/m
        V_SCM = m*SCM/trace(SCM);

        err_SCM = V_SCM(:)-theta_true;
        MSE_SCM = MSE_SCM + (err_SCM'*err_SCM)/Ns;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tyler's estimator
        Ty = Tyler_C_11(y, Max_it_Ty);

        %%%%%% For S(Sigma) = trace(Sigma)/m
        V_Ty = m*Ty/trace(Ty);

        err_Ty = V_Ty(:)-theta_true;
        MSE_Ty= MSE_Ty + (err_Ty'*err_Ty)/Ns;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constrained R-estimator with the van der Waerden score function
        [V_RvdW, a] = R_vdW_est_const_trace(y, V_Ty, perturbation_par);

        err_RvdW = V_RvdW(:)-theta_true;
        MSE_R_vdW = MSE_R_vdW + (err_RvdW'*err_RvdW)/Ns;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constrained R-estimator with the t_based score function
        [V_Rt, a] = R_t_nu_est_const_trace(y, V_Ty, nu_par, perturbation_par);

        err_Rt = V_Rt(:)-theta_true;
        MSE_R_t_nu = MSE_R_t_nu + (err_Rt'*err_Rt)/Ns;

    end
    Fro_MSE_SCM(il) = MSE_SCM;
    Fro_MSE_Ty(il) = MSE_Ty;
    Fro_MSE_R_vdW(il) = MSE_R_vdW;
    Fro_MSE_R_t_nu(il) = MSE_R_t_nu;

    a1 = -1/(m+(nu/2)+1);
    a2 = (nu/2 + m)/(m+(nu/2)+1);
    
    % FIM
    FIM_Sigma = n * (a1*(Inv_V_S(:)*Inv_V_S(:)') + a2*kron(Inv_V_S.',Inv_V_S));
    
    CRB = U*inv(U'*FIM_Sigma*U)*U';
    
    SFIM_Sigma = n * a2*(kron(Inv_V_S.',Inv_V_S) - (1/m)*(Inv_V_S(:)*Inv_V_S(:)'));
    
    SCRB = U*inv(U'*SFIM_Sigma*U)*U';


    SCR_Bound(il) = trace(real(SCRB));
    CR_Bound(il) = trace(real(CRB));
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
title('Estimation of $\mathrm{vec}(\mathbf{V}_S)$ with $S(\mathbf{\Sigma})=\mathrm{trace}(\mathbf{\Sigma})/m$','interpreter','latex')
