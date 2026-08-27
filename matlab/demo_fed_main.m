function demo_fed_main(varargin)
%DEMO_FED_MAIN  Minimal FED demonstration (main release).
%
%   Builds the standard synthetic test signal, runs fed_cov_main for:
%     (1) F = I  (SSA baseline)
%     (2) F = HFD^2, p = 2  (paper default)
%   Prints reconstruction metrics and saves an optional figure.
%
%   Usage (from this folder):
%     demo_fed_main
%     demo_fed_main('SaveFig', false)
%
%   See also: fed_cov_main, verify_fed_main

here = fileparts(mfilename('fullpath'));
addpath(here);

ip = inputParser;
addParameter(ip, 'Fs',       128);
addParameter(ip, 'Duration', 4);
addParameter(ip, 'L',        48);
addParameter(ip, 'NumModes', 12);
addParameter(ip, 'Seed',     42);
addParameter(ip, 'SaveFig',  true);
addParameter(ip, 'OutDir',   fullfile(here, 'output'));
parse(ip, varargin{:});
p = ip.Results;

rng(p.Seed);
T = p.Fs * p.Duration;
t = (0:T-1)' / p.Fs;

fprintf('\n=== FED main demo ===\n');
fprintf('Fs=%d Hz, N=%d, L=%d, k=%d\n\n', p.Fs, T, p.L, p.NumModes);

%% Synthetic signal (Section 5.1 style)
c1 = 1.0 * sin(2*pi*3*t);
c2 = 0.7 * sin(2*pi*12*t) .* (0.6 + 0.5*sin(2*pi*0.7*t));
c3 = zeros(T, 1);
for bt = [0.5, 1.5, 2.5, 3.5]
    c3 = c3 + 0.5 * exp(-((t-bt).^2)/(2*0.1^2)) .* sin(2*pi*30*t);
end
x = c1 + c2 + c3 + 0.1 * randn(T, 1);

common = {'L', p.L, 'NumModes', p.NumModes, 'FStrategy', 'row', ...
    'Kmax', 6, 'Verbose', true};

fprintf('--- SSA baseline (F = I) ---\n');
out_ssa = fed_cov_main(x, common{:}, 'FTransform', 'identity', 'FParam', 1);

fprintf('\n--- FED default (F = HFD^2, p = 2) ---\n');
out_fed = fed_cov_main(x, common{:}, 'FTransform', 'power', 'FParam', 2);

fprintf('\n=== Summary ===\n');
fprintf('%-22s  epsL2   epsF_high   F-Sel   Finv_orth\n', 'Method');
fprintf('%s\n', repmat('-', 1, 58));
print_row('SSA (F=I)', out_ssa);
print_row('FED (HFD^2,p=2)', out_fed);

if p.SaveFig
    if ~isfolder(p.OutDir), mkdir(p.OutDir); end
    fig = figure('Color', 'w', 'Position', [100 100 1100 420], 'Visible', 'off');
    subplot(1, 2, 1);
    plot(t, x, 'Color', [0.2 0.2 0.2]); hold on;
    plot(t, out_ssa.x_hat, 'b', 'LineWidth', 1.1);
    grid on; xlabel('Time (s)'); ylabel('Amplitude');
    title(sprintf('SSA (F=I)  \\epsilon_{L2}=%.3f', out_ssa.epsilon_L2));
    legend('Signal', 'Reconstruction', 'Location', 'best');

    subplot(1, 2, 2);
    plot(t, x, 'Color', [0.2 0.2 0.2]); hold on;
    plot(t, out_fed.x_hat, 'Color', [0 0.45 0.74], 'LineWidth', 1.1);
    grid on; xlabel('Time (s)'); ylabel('Amplitude');
    title(sprintf('FED (HFD^2)  \\epsilon_{L2}=%.3f', out_fed.epsilon_L2));
    legend('Signal', 'Reconstruction', 'Location', 'best');

    fpath = fullfile(p.OutDir, 'demo_fed_main_reconstruction.png');
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, fpath, 'Resolution', 150);
    else
        print(fig, fpath, '-dpng', '-r150');
    end
    close(fig);
    fprintf('\nFigure saved: %s\n', fpath);
end

fprintf('\nDone. Run verify_fed_main for the Step-6 algebra check.\n');
end

function print_row(name, o)
fprintf('%-22s  %5.3f   %9.3f   %5.3f   %8.1e\n', ...
    name, o.epsilon_L2, o.epsilon_F_local, o.F_selectivity, ...
    o.Finv_orthogonality_deviation);
end
