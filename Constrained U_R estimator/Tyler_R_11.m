function [C, iter] = Tyler_R_11(y, MAX_ITER)
%
% Author:         
% Esa Ollila, esa.ollila@aalto.fi

[N K] = size(y);

EPS = 1.0e-4;   % Iteration accuracy

invC0 = eye(N);  
iter = 1;

z=y.';
while (iter<MAX_ITER)     
      s = real(sum(conj(z)*invC0.*z,2));
      w = N./s;
      C = z.'*(z.*repmat(w,1,N))/K;
      C=C/C(1,1);
      d = norm(eye(N)-invC0*C,1);
      if (d<=EPS) break; end
      invC0 = C\eye(N);
      iter = iter+1;     
end 
