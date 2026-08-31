function nf_proposed(mode)
% NF_PROPOSED  The proposed three-probe near-field localization scheme.
%
%   nf_proposed            run a single-point demo
%   nf_proposed('sweepN')  RMSE vs number of antennas
%   nf_proposed('sweepW')  RMSE vs bandwidth
%
% MATLAB counterpart of sim/exp_f_three_probe.py and sim/exp_k_sweeps.py, so
% the sweeps can be run at publication scale offline.  Raise N_TRIAL below.
%
% THE SCHEME, T = 3 probes
%   Stage A (1 probe)   a beam-squint sweep gives the coarse ANGLE.  Its range
%                       output is discarded -- see Experiment B, where the
%                       one-dimensional trajectory covers 1.3% of the region.
%                       Here it is modelled by its measured error statistic.
%   Stage B (2 probes)  a constant-modulus monopulse pair straddling that angle
%                       by 0.25 beamwidths, with the range guess pinned at 30 m
%                       and never estimated.
%
% THE ESTIMATOR
%   Step 1  range from the phase slope of the sum channel across frequency.
%           This is the whole point: the received phase is exactly linear in
%           subcarrier index with slope proportional to the per-antenna
%           propagation distance, with NO near-field approximation.
%   Step 2  Gauss-Newton on the whitened likelihood, amplitude concentrated out.
%
% Two details that silently break it if got wrong, both found the hard way:
%   * the coarse delay must be read as SIGNED -- searching only the lower half
%     of the transform returns a wrong range whenever r < r_guess;
%   * the monopulse pair must NOT be orthonormalised -- QR mixes the two beams
%     and destroys the sum channel the delay slope is read from.  The whitener
%     handles the non-orthogonality instead.

if nargin < 1, mode = 'demo'; end
switch mode
    case 'demo',   demo();
    case 'sweepN', sweepN();
    case 'sweepW', sweepW();
    otherwise, error('unknown mode %s', mode);
end
end

% ------------------------------------------------------------------ drivers

function demo()
cfg = config(); rng(0);
r_true = 30; th_true = deg2rad(15);
bw = 0.886*cfg.lambda/cfg.D;
fprintf('N=%d  M=%d  W=%.0f GHz  beamwidth=%.4f deg\n', cfg.N, cfg.M, cfg.W/1e9, rad2deg(bw));
fprintf('unambiguous range c*M/(2W) = %.1f m\n\n', cfg.c0*cfg.M/(2*cfg.W));
fprintf('%6s%14s%14s\n','SNR','RMSE_th(deg)','RMSE_r(m)');
for snr = [-10 -5 0 5 10 15 20]
    [rt, rr] = trial_set(cfg, r_true, th_true, snr, 30, 0.153*bw, 0.25*bw);
    fprintf('%6d%14.3e%14.3e\n', snr, rt, rr);
end
end

function sweepN()
Ns = [64 128 256 512 1024]; snr = 0; N_TRIAL = 15;   % raise for publication
r_true = 30; th_true = deg2rad(15);
res = zeros(numel(Ns),2);
for i = 1:numel(Ns)
    cfg = config(Ns(i)); bw = 0.886*cfg.lambda/cfg.D; rng(i);
    [res(i,1), res(i,2)] = trial_set(cfg, r_true, th_true, snr, N_TRIAL, 0.153*bw, 0.25*bw);
    fprintf('N=%5d  RMSE_th %.3e deg  RMSE_r %.3e m\n', Ns(i), res(i,1), res(i,2));
end
p = polyfit(log(Ns(:)), log(res(:,2)), 1);
fprintf('\nfitted exponent in N: %+.2f   predicted -0.50 (curvature would be -2.50)\n', p(1));
figure; loglog(Ns, res(:,2), 'o-', 'LineWidth',1.8); hold on;
loglog(Ns, res(1,2)*(Ns/Ns(1)).^-0.5, '--'); grid on;
xlabel('number of antennas N'); ylabel('RMSE_r (m)');
legend('measured','N^{-1/2} reference');
end

function sweepW()
Ws = [0.5 1 2 3 6]*1e9; snr = 0; N_TRIAL = 15;      % raise for publication
r_true = 30; th_true = deg2rad(15);
res = zeros(numel(Ws),2);
for i = 1:numel(Ws)
    cfg = config(256, Ws(i)); bw = 0.886*cfg.lambda/cfg.D; rng(100+i);
    % c*M/(2W) must stay above the sensing region or the delay wraps
    assert(cfg.c0*cfg.M/(2*cfg.W) > 50, 'bandwidth too large: range would alias');
    [res(i,1), res(i,2)] = trial_set(cfg, r_true, th_true, snr, N_TRIAL, 0.153*bw, 0.25*bw);
    fprintf('W=%4.1f GHz  RMSE_th %.3e deg  RMSE_r %.3e m\n', Ws(i)/1e9, res(i,1), res(i,2));
end
p = polyfit(log(Ws(:)), log(res(:,2)), 1);
fprintf('\nfitted exponent in W: %+.2f   predicted -1.00 (curvature is blind to W)\n', p(1));
figure; loglog(Ws/1e9, res(:,2), 's-', 'LineWidth',1.8); hold on;
loglog(Ws/1e9, res(1,2)*(Ws/Ws(1)).^-1, '--'); grid on;
xlabel('bandwidth W (GHz)'); ylabel('RMSE_r (m)');
legend('measured','W^{-1} reference');
end

function [rmse_th, rmse_r] = trial_set(cfg, r_true, th_true, snr_db, n_trial, ce, delta)
et = zeros(n_trial,1); er = zeros(n_trial,1);
for t = 1:n_trial
    th0 = th_true + ce*randn;
    V = beams(cfg, 30, th0, delta);
    z = simulate(cfg, r_true, th_true, V, snr_db);
    [th, r] = estimate(cfg, z, V, th0, 30, 3*ce);
    et(t) = th - th_true;  er(t) = r - r_true;
end
rmse_th = rad2deg(sqrt(mean(et.^2)));  rmse_r = sqrt(mean(er.^2));
end

% ------------------------------------------------------------------ the scheme

function V = beams(cfg, r0, th0, delta)
% Constant-modulus monopulse pair.  Focusing every subcarrier on one point takes
%   phi_n = f0*r_n/c   (phase shifters),   t_n = r_n/c   (true-time delays),
% giving v_m(n) = exp(-j2pi f_m r_n/c): unit modulus, no amplitude taper.
% NOT orthonormalised -- see the header.
V = zeros(numel(cfg.f), cfg.N, 2);
tt = [th0-delta, th0+delta];
for k = 1:2
    d = dist_fresnel(r0, tt(k), cfg.x);
    V(:,:,k) = exp(-2i*pi*cfg.f(:)*d(:).'/cfg.c0)/sqrt(cfg.N);
end
end

function h = response(cfg, r, th, V)
a = exp(-2i*pi*cfg.f(:)*dist_fresnel(r,th,cfg.x)/cfg.c0);
h = squeeze(sum(conj(V).*a, 2));
end

function z = simulate(cfg, r, th, V, snr_db)
% Noise referred to the antennas, then combined.
a = exp(-2i*pi*cfg.f(:)*dist_fresnel(r,th,cfg.x)/cfg.c0) * exp(2i*pi*rand);
s = sqrt(10^(-snr_db/10)/2);
n = s*(randn(size(a)) + 1i*randn(size(a)));
z = squeeze(sum(conj(V).*(a+n), 2));
end

function Wm = whitener(V)
% Combined noise covariance is sigma^2 V^H V; whiten by its Cholesky factor.
G = squeeze(sum(conj(permute(V,[1 2 3])).*permute(V,[1 2 3]),2));  %#ok<NASGU>
T = size(V,3); G = zeros(T);
for i = 1:T
    for j = 1:T
        G(i,j) = sum(sum(conj(V(:,:,i)).*V(:,:,j)))/size(V,1);
    end
end
Wm = inv(chol(G + 1e-12*eye(T), 'lower'))';
end

function r = coarse_range(z, cfg, r0)
% SIGNED offset from the beam focus.  A user closer than the focus puts the peak
% in the upper half of the transform; searching only the lower half is wrong.
s = sum(z, 2);
df = cfg.f(2)-cfg.f(1);
nfft = 2^nextpow2(numel(cfg.f)*8);
prof = fft(conj(s), nfft);
[~, k] = max(abs(prof));  k = k - 1;
if k >= nfft/2, k = k - nfft; end
r = r0 + k*cfg.c0/(nfft*df);
end

function [th_hat, r_hat] = estimate(cfg, z, V, th_init, r0, th_halfwidth)
r_try = coarse_range(z, cfg, r0);
r_try = min(max(r_try, 5), 80);
Wm = whitener(V);
zw = z*Wm.';  zw = zw(:);

    function res = resid(p)
        h = response(cfg, p(1), p(2), V)*Wm.';  h = h(:);
        alpha = (h'*zw)/max(real(h'*h), 1e-300);
        e = zw - alpha*h;
        res = [real(e); imag(e)];
    end

lo = [max(0.5, r_try-5), th_init-th_halfwidth];
hi = [r_try+5,           th_init+th_halfwidth];
opts = optimoptions('lsqnonlin','Display','off','FunctionTolerance',1e-14, ...
                    'StepTolerance',1e-14,'OptimalityTolerance',1e-14);
p = lsqnonlin(@resid, [r_try, th_init], lo, hi, opts);
r_hat = p(1); th_hat = p(2);
end

% ------------------------------------------------------------------ model

function c = config(N, W)
if nargin < 1, N = 256; end
if nargin < 2, W = 3e9; end
c.c0 = 299792458; c.N = N; c.fc = 60e9; c.W = W; c.M = 2048;
c.lambda = c.c0/c.fc; c.d = c.lambda/2; c.f0 = c.fc - W/2;
c.D = (N-1)*c.d;
c.x = ((0:N-1) - (N-1)/2)*c.d;
c.f = c.f0 + (0:2:c.M-1)*(W/c.M);      % every 2nd tone, as in the Python
end

function d = dist_fresnel(r, th, x)
d = r - x*sin(th) + x.^2*cos(th)^2/(2*r);
end
