% Introduce variables
% gi is propagation constant in ith region
% Zi is characteristic impedance of its region
% Li is the length of region i
syms g1 Z1 L1 g2 Z2 L2 g3 Z3 L3
num_segments = 3;
identity = eye(num_segments*2);
% Define V and I for each region.
% You can also regard the cascaded middle region as one region if you
% want...
V1_hat = identity(1,:);
I1_hat = identity(2,:);
V2_hat = identity(3,:);
I2_hat = identity(4,:);
V3_hat = identity(5,:);
I3_hat = identity(6,:);
% We derive the ABCD matrix of middle segment.
Zs_2 = g2*Z2; % This is the series impedance, aka R+iwL
Ys_2 = g2/Z2; % This is the shunt admittance, aka G+iwC
G2 = [0 -Zs_2;-Ys_2 0]; % This is the generator of translation
[V2,D2] = eig(G2);% This gives us the eigenvectors V and eigenvalues D of G
% Test that we can diagonalize G
% simplify(inv(V2)*G2*V2)
% Compute the matrix exponential - note that the zero elements need to stay
% zero
off_diag_ones = (1+zeros(2,2)) - eye(2);
expG2L2 = V2 * (exp(D2*L2) - off_diag_ones) * inv(V2);
% Now we propagate V2 and I2 to position L
V2I2_L = expG2L2 * [V2_hat;I2_hat];
% Now we get V2_L_hat and I2_L_hat
V2_L_hat = V2I2_L(1,:);
I2_L_hat = V2I2_L(2,:);
% Now we can construct our a and b wave operators:
Z_ref_1 = Z1; % can make this be 50 if we want
Z_ref_2 = Z3; % can make this be 50 if we want
a1_hat = V1_hat + I1_hat*Z_ref_1;
b1_hat = V1_hat - I1_hat*Z_ref_1;
a2_hat = V3_hat - I3_hat*Z_ref_2;
b2_hat = V3_hat + I3_hat*Z_ref_2;
% we construct boundary condition matrix
BC1 = a1_hat;
BC2 = a2_hat;
BC3 = V1_hat - V2_hat;
BC4 = I1_hat - I2_hat;
BC5 = V2_L_hat - V3_hat;
BC6 = I2_L_hat - I3_hat;
BC = [BC1;BC2;BC3;BC4;BC5;BC6];
% We now compute S11 and S21
solution_vector_1 = [1 0 0 0 0 0]';
psi_1 = inv(BC)*solution_vector_1;
S11 = b1_hat*psi_1
S21 = b2_hat*psi_1
% We now compute S22 and S12
solution_vector_2 = [0 1 0 0 0 0]';
psi_2 = inv(BC)*solution_vector_2;
S22 = b2_hat*psi_2
S12 = b1_hat*psi_2