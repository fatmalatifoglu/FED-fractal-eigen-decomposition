function verify_fed_main()
%VERIFY_FED_MAIN  Quick check: Step 6 uses v = F^(1/2) y (not F^(-1/2) y).
%
%   Test 1 — random C, F: eigenvalues and ||Cv - lambda F^(-1) v|| ~ 0
%   Test 2 — fed_cov_main on a short signal: orthogonality & residual fields
%
%   Usage:
%     verify_fed_main

here = fileparts(mfilename('fullpath'));
addpath(here);

fprintf('=== FED main verification ===\n\n');
rng(42);

%% Test 1: algebra
L = 6;
A = randn(L, L);
C = (A * A' + A * A') / 2;
F_diag = [1.0, 2.5, 0.7, 3.2, 1.8, 0.4]';
Finv = diag(1 ./ F_diag);
Fh = diag(sqrt(F_diag));

[V_gt, D_gt] = eig(C, Finv);
lam_gt = sort(real(diag(D_gt)), 'descend');

M = Fh * C * Fh; M = (M + M') / 2;
[Y, Lam] = eig(M);
lam = sort(real(diag(Lam)), 'descend');
V = Fh * Y;

res = 0;
for i = 1:L
    res = max(res, norm(C*V(:,i) - lam(i)*(Finv*V(:,i))));
end
orth = norm(V' * Finv * V - eye(L), 'fro');

fprintf('Test 1 (random C,F):\n');
fprintf('  eigenvalue diff:     %.2e\n', norm(lam_gt - lam));
fprintf('  max Cv residual:     %.2e  (target ~0)\n', res);
fprintf('  Finv orth deviation: %.2e  (target ~1e-14)\n', orth);

V_old = diag(1./sqrt(F_diag)) * Y;
res_old = 0;
for i = 1:L
    res_old = max(res_old, norm(C*V_old(:,i) - lam(i)*(Finv*V_old(:,i))));
end
fprintf('  (old F^(-1/2)Y residual for comparison: %.4f)\n\n', res_old);

%% Test 2: fed_cov_main output fields
x = randn(256, 1);
out = fed_cov_main(x, 'L', 48, 'NumModes', 8, 'FTransform', 'power', ...
    'FParam', 2, 'Verbose', false);

fprintf('Test 2 (fed_cov_main):\n');
fprintf('  Finv_orthogonality_deviation: %.2e\n', out.Finv_orthogonality_deviation);
fprintf('  cv_eq_residual:               %.2e\n', out.cv_eq_residual);
fprintf('  formula: %s\n', out.formula);

ok = res < 1e-8 && orth < 1e-6 && out.Finv_orthogonality_deviation < 1e-6;
if ok
    fprintf('\nPASS — main release implementation is consistent.\n');
else
    fprintf('\nWARN — check tolerances or implementation.\n');
end
end
