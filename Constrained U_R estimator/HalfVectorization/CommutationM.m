function [K] = CommutationM(A)

[m, n] = size(A);
I = reshape(1:m*n, [m, n]); % initialize a matrix of indices of size(A)
I = I'; % Transpose it
I = I(:); % vectorize the required indices
Y = eye(m*n); % Initialize an identity matrix
K = Y(I,:); % Re-arrange the rows of the identity matrix
end

