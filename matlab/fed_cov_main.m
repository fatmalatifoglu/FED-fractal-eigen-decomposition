function out = fed_cov_main(x, varargin)
% FED_COV_MAIN  Fractal Eigen Decomposition (covariance form, main release)
%
%   Generalized eigenproblem:  C v = lambda * F^(-1) v
%   Equivalent symmetric form: M y = lambda y,  M = F^(1/2) C F^(1/2)
%   Back-projection (Step 6):    v = F^(1/2) y
%   F^(-1)-normalization:        v_i <- v_i / sqrt(v_i' F^(-1) v_i)
%
%   When F = I this reduces to standard SSA (Proposition 4).
%
%   INTERPRETATION OF F:
%     F_ii = HFD(row i)   <- larger value = more important region
%     High HFD -> large lambda -> mode is prioritized
%
%   THEORETICAL BASIS (correct derivation):
%     C v = lambda F^(-1) v
%     Substitute v = F^(1/2) y  (y = F^(-1/2) v):
%     C F^(1/2) y = lambda F^(-1) F^(1/2) y = lambda F^(-1/2) y
%     Multiply both sides on the left by F^(1/2):
%     F^(1/2) C F^(1/2) y = lambda F^(1/2) F^(-1/2) y = lambda y
%     => M y = lambda y,  M = F^(1/2) C F^(1/2),  v = F^(1/2) y
%
%   Verification: v^T F^(-1) v = (F^(1/2)y)^T F^(-1) (F^(1/2)y)
%                            = y^T F^(1/2) F^(-1) F^(1/2) y
%                            = y^T y = I   (Y is Euclidean-orthonormal)
%   => v = F^(1/2) y is automatically F^(-1)-orthonormal; the extra
%     normalization step below is kept only for numerical safety (~no-op).
%
%   lambda = (v^T C v) / (v^T F^(-1) v)
%
%   F is diagonal: F_ii from Higuchi FD of trajectory row i (transform: power, exp, ...).
%   High-HFD rows receive larger eigenvalues under F^(-1) weighting.

ip = inputParser;
addRequired(ip,  'x');
addParameter(ip, 'L',           []);
addParameter(ip, 'NumModes',    12);
addParameter(ip, 'FStrategy',   'row');
addParameter(ip, 'FTransform',  'power');
addParameter(ip, 'FParam',      2);
addParameter(ip, 'Kmax',        6);
addParameter(ip, 'Verbose',     false);
parse(ip, x, varargin{:});
prm = ip.Results;

x = double(x(:));
N = length(x);

if isempty(prm.L), L = floor(N/3); else, L = prm.L; end
L = min(L, floor(N/2));
K = N - L + 1;

%% ── Hankel embedding ────────────────────────────────────────────────
X_traj = zeros(L, K);
for i = 1:K
    X_traj(:, i) = x(i:i+L-1);
end

%% ── Covariance matrix ───────────────────────────────────────────────
C = (X_traj * X_traj') / K;
C = (C + C') / 2;

%% ── Raw fractal metric ──────────────────────────────────────────────
HFD_raw = zeros(L, 1);
switch lower(prm.FStrategy)
    case 'row'
        for i = 1:L
            row_data = X_traj(i, :)';
            if K >= max(8, 2*prm.Kmax)
                HFD_raw(i) = higuchi_fd(row_data, prm.Kmax);
            else
                HFD_raw(i) = std(row_data);
            end
        end
    case 'seg'
        for i = 1:L
            seg = x(i:i+K-1);
            if length(seg) >= max(8, 2*prm.Kmax)
                HFD_raw(i) = higuchi_fd(seg, prm.Kmax);
            else
                HFD_raw(i) = std(seg);
            end
        end
    case 'none'
        HFD_raw = ones(L, 1);
end

%% ── F transformation ────────────────────────────────────────────────
F_diag = transform_F(HFD_raw, prm.FTransform, prm.FParam);
F_diag = max(F_diag, 1e-6);

if prm.Verbose
    fprintf('  [FED-COV main] N=%d, L=%d, K=%d\n', N, L, K);
    fprintf('  [FED-COV main] F: min=%.3f, max=%.3f, ratio=%.2f\n', ...
        min(F_diag), max(F_diag), max(F_diag)/min(F_diag));
end

% Diagonal inverse for F^(-1)
F_inv_diag = 1 ./ F_diag;
F_inv = diag(F_inv_diag);

%% ── Solve: C v = lambda F^(-1) v ────────────────────────────────────
% Equivalent: F^(1/2) C F^(1/2) y = lambda y, v = F^(1/2) y  (see header)
% Symmetric eigenproblem -> numerically stable

F_half = diag(sqrt(F_diag));

% Symmetric reduction: M = F^(1/2) C F^(1/2)
M = F_half * C * F_half;
M = (M + M') / 2;

[Y, Lam] = eig(M);
lams = real(diag(Lam));
[lams, oo] = sort(lams, 'descend');
Y = real(Y(:, oo));

k_use = min(prm.NumModes, L - 1);
lams = lams(1:k_use);
Y    = Y(:, 1:k_use);

% Back-projection: v = F^(1/2) y
V = F_half * Y;

% F^(-1)-normalization: v^T F^(-1) v = 1
% With the correct substitution this is already satisfied analytically
% (v = F^(1/2)y => v^T F^(-1) v = y^T y = I, since Y is Euclidean-orthonormal).
% The loop below is kept only for floating-point safety (~no-op).
for i = 1:k_use
    nrm = sqrt(max(V(:,i)' * F_inv * V(:,i), eps));
    V(:,i) = V(:,i) / nrm;
end

%% ── Euclidean normalization for reconstruction ───────────────────────
% v_i is F^(-1)-normalized but SSA-style reconstruction expects Euclidean modes.
% NOTE: when F != I, U is NOT required to be Euclidean-orthogonal;
% Proposition 3 guarantees F^(-1)-orthogonality only (see Section 6.6).
U = zeros(L, k_use);
for i = 1:k_use
    nrm = norm(V(:, i)) + eps;
    U(:, i) = V(:, i) / nrm;
end

%% ── SSA-style reconstruction ────────────────────────────────────────
components = zeros(N, k_use);
sigmas     = zeros(k_use, 1);
energies   = zeros(k_use, 1);

for i = 1:k_use
    u_i = U(:, i);
    var_i = u_i' * C * u_i;
    sigma_i = sqrt(max(K * var_i, 0));
    sigmas(i) = sigma_i;

    if sigma_i < eps, continue; end

    V_traj_i = (X_traj' * u_i) / sigma_i;
    X_i      = sigma_i * u_i * V_traj_i';

    components(:, i) = antidiag_avg(X_i, N);
    energies(i)      = sum(components(:, i).^2);
end

x_hat   = sum(components, 2);
residue = x - x_hat;

%% ── Error metrics ───────────────────────────────────────────────────
epsilon_L2 = norm(residue) / (norm(x) + eps);

R_traj = zeros(L, K);
for i = 1:K, R_traj(:, i) = residue(i:i+L-1); end

% F-weighted global error (natural F: high HFD = important)
% Errors in important regions are penalized more heavily
res_F_norm = sqrt(trace(R_traj' * F_half * F_half * R_traj));
sig_F_norm = sqrt(trace(X_traj' * F_half * F_half * X_traj));
epsilon_F_global = res_F_norm / (sig_F_norm + eps);

% Regional errors
HFD_thr = quantile(HFD_raw, 0.75);
high_idx = HFD_raw > HFD_thr;

if sum(high_idx) > 0 && sum(high_idx) < L
    res_high = R_traj(high_idx, :);
    sig_high = X_traj(high_idx, :);
    epsilon_F_local = norm(res_high, 'fro') / (norm(sig_high, 'fro') + eps);

    res_low = R_traj(~high_idx, :);
    sig_low = X_traj(~high_idx, :);
    epsilon_F_low = norm(res_low, 'fro') / (norm(sig_low, 'fro') + eps);
else
    epsilon_F_local = epsilon_L2;
    epsilon_F_low   = epsilon_L2;
end

% F-Selectivity: delay positions where the strongest modes are active
mod_HFD_active = zeros(k_use, 1);
for i = 1:k_use
    weights = abs(U(:,i));
    if sum(weights) > eps
        mod_HFD_active(i) = sum(weights .* HFD_raw) / sum(weights);
    end
end
if k_use >= 3
    F_selectivity = corr(energies, mod_HFD_active);
else
    F_selectivity = NaN;
end

%% ── Internal verification metrics ───────────────────────────────────
% Confirm correct Step-6 implementation (Finv_orth_dev ~ 1e-14 for F=I and F!=I).
Finv_orth_dev = norm(V' * F_inv * V - eye(k_use), 'fro');
% Residual of C v - lambda F^(-1) v (first mode)
cv_residual = norm(C*V(:,1) - lams(1)*(F_inv*V(:,1)));

%% ── Output ──────────────────────────────────────────────────────────
out.components    = components;
out.modes_2d_F    = V;
out.modes_2d_E    = U;
out.lambda        = lams;
out.sigmas        = sigmas;
out.energies      = energies;
out.HFD_raw       = HFD_raw;
out.F_diag        = F_diag;
out.F_inv_diag    = F_inv_diag;
out.X_traj        = X_traj;
out.C             = C;
out.x_orig        = x;
out.x_hat         = x_hat;
out.residue       = residue;

out.epsilon_L2        = epsilon_L2;
out.epsilon_F_global  = epsilon_F_global;
out.epsilon_F_local   = epsilon_F_local;   % high-HFD region
out.epsilon_F_low     = epsilon_F_low;     % low-HFD region
out.F_selectivity     = F_selectivity;
out.mod_HFD_active    = mod_HFD_active;

out.Finv_orthogonality_deviation = Finv_orth_dev;   % expect ~1e-14
out.cv_eq_residual               = cv_residual;      % expect ~0

out.L = L; out.K = K; out.N = N; out.k = k_use;
out.params = prm;
out.formula = 'C v = lambda * F^(-1) v  [v = F^(1/2) y]';

if prm.Verbose
    fprintf('  [FED-COV main] epsL2=%.4f | epsF_g=%.4f | epsF_y=%.4f | epsF_d=%.4f\n', ...
        epsilon_L2, epsilon_F_global, epsilon_F_local, epsilon_F_low);
    fprintf('  [FED-COV main] F_selectivity=%.3f | lambda_1=%.3f\n', F_selectivity, lams(1));
    fprintf('  [FED-COV main] check: Finv_orth_dev=%.2e | cv_residual=%.2e\n', ...
        Finv_orth_dev, cv_residual);
end
end

%% ─────────────────────────────────────────────────────────────────────
function f_out = transform_F(HFD, transform, param)
hfd_n = HFD - min(HFD);
if max(hfd_n) > eps, hfd_n = hfd_n / max(hfd_n); end

switch lower(transform)
    case 'identity'
        f_out = ones(size(HFD));
    case 'linear'
        f_out = 0.1 + 0.9 * hfd_n;
    case 'power'
        f_out = 0.1 + 0.9 * (hfd_n .^ param);
    case 'exp'
        f_out = exp(param * hfd_n);
        f_out = f_out / max(f_out);
        f_out = max(f_out, 0.01);
    case 'threshold'
        thr = quantile(HFD, param);
        f_out = ones(size(HFD)) * 0.1;
        f_out(HFD > thr) = 1.0;
    case 'sigmoid'
        z = (HFD - mean(HFD)) / (std(HFD) + eps);
        f_out = 1 ./ (1 + exp(-param * z));
        f_out = max(f_out, 0.01);
end
f_out = max(f_out, 1e-6);
end

function y = antidiag_avg(X, N)
[L, K] = size(X);
y = zeros(N, 1); c = zeros(N, 1);
for i = 1:L
    for j = 1:K
        n = i + j - 1;
        if n >= 1 && n <= N
            y(n) = y(n) + X(i, j);
            c(n) = c(n) + 1;
        end
    end
end
y = y ./ max(c, 1);
end

function fd = higuchi_fd(x, kmax)
x = double(x(:)); N = length(x);
kmax = min(kmax, max(1, floor(N/3)));
if N < 8 || kmax < 2, fd = std(x); return; end
L = zeros(1, kmax);
for k = 1:kmax
    Lk = 0;
    for m = 1:k
        idx = m:k:N;
        Nm = length(idx) - 1;
        if Nm < 1, continue; end
        denom = Nm * k^2 / (N-1);
        if denom < eps, continue; end
        Lk = Lk + sum(abs(diff(x(idx)))) / denom;
    end
    L(k) = Lk / k;
end
valid = L > eps;
if sum(valid) < 2, fd = std(x); return; end
ks = (1:kmax)';
p = polyfit(log(ks(valid)), log(L(valid)' + eps), 1);
fd = max(min(-p(1), 2), 0);
if isnan(fd) || isinf(fd), fd = std(x); end
end
