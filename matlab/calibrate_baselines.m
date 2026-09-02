function calibrate_baselines(mode)
% CALIBRATE_BASELINES  Recover the parameters the source papers never state,
% by matching the numbers those papers publish.
%
%   calibrate_baselines('stagger')   <-- RUN THIS FIRST.  Fits Luo2024's sweep
%                                        stagger to its own Fig. 13 anchors.
%   calibrate_baselines('rmid')          Fits Luo2024's r_mid1 / r_mid2.
%   calibrate_baselines('grid')          Confirms the search grid is fine enough.
%   calibrate_baselines('S')             Sweeps Zhang2026's |S|.
%   calibrate_baselines('all')           All four in order.
%
% WHY THIS IS NEEDED
%   Both papers leave parameters unspecified that decide their own results.
%   Luo2024 never states the sweep stagger, and holding everything else fixed it
%   swings the range RMSE over three orders of magnitude.  Zhang2026 defines |S|
%   in the text but omits it from its Table II, and the achievable accuracy
%   scales as 1/sqrt(|S|).  A comparison built on guessed values is not a
%   comparison, so each one is fitted to a number its own paper publishes.
%
% THE ANCHORS  (Luo2024 Sec. V-C, its Fig. 13)
%   N = 128, M+1 = 2048, W = 1 GHz, user at (60 m, 20 deg), region 5-80 m,
%   +-60 deg, SNR = 10 dB:
%       P = 4   ->  RMSE_theta = 0.031 deg,  RMSE_r = 0.165 m
%       P = 12  ->  RMSE_theta = 0.018 deg,  RMSE_r = 0.037 m
%
% ONE PARAMETER CANNOT BE FITTED THIS WAY, and that matters.
%   Zhang2026's anchor is its claim of RMSE_r < 0.001 m.  Even at |S| = M, the
%   incoherent bound of its own algorithm class is 4.91e-3 m at 20 dB, so NO
%   value of |S| reaches the claim.  Mode 'S' therefore does not fit |S|; it
%   shows the trend against the bound so the largest tractable |S| can be used
%   and reported as the most generous possible assumption.
%
% HOW TO USE THE OUTPUT
%   Each mode prints a table and names the value minimising the mismatch against
%   the anchors.  Put that value into the CFG block below, then re-run the
%   comparison scripts.  Raise N_TRIAL for a tighter fit; 60 is enough to rank
%   candidates, 500 to quote a value.

if nargin < 1, mode = 'all'; end

% ===================== EDIT THESE AFTER EACH CALIBRATION =====================
CFG.stagger_deg = 4.0;     % <- set from mode 'stagger'
CFG.stagger_mode = 'total';% 'total' = the P sweeps divide a fixed angular span,
                           % so the span does not grow with P and only sqrt(P)
                           % averaging applies.
                           % 'step'  = a fixed angular STEP between consecutive
                           % sweeps, so the span grows as (P-1) and the peak
                           % subcarriers spread further as P rises.
                           % Luo2024 says only that the spans differ "slightly",
                           % which does not distinguish the two -- and they scale
                           % with P completely differently.  Mode 'stagger' now
                           % fits both and reports which reproduces the paper.
CFG.rmid_frac   = 0.5;     % <- set from mode 'rmid'  (fraction across the region)
CFG.n_grid      = 4000;    % <- set from mode 'grid'
CFG.S_sub       = 64;      % <- set from mode 'S'
CFG.N_TRIAL     = 60;      % raise to 500 to quote a value
% =============================================================================

switch mode
    case 'stagger', fit_stagger(CFG);
    case 'rmid',    fit_rmid(CFG);
    case 'grid',    fit_grid(CFG);
    case 'S',       sweep_S(CFG);
    case 'all',     fit_stagger(CFG); fit_rmid(CFG); fit_grid(CFG); sweep_S(CFG);
    otherwise, error('unknown mode %s', mode);
end
end

% ============================================================ calibration modes

function fit_stagger(CFG)
fprintf('\n=== 1. Luo2024 sweep stagger, against its Fig. 13 anchors ===\n');
fprintf('Anchors: P=4 -> 0.031 deg / 0.165 m,  P=12 -> 0.018 deg / 0.037 m\n\n');
fprintf('The angle turns out to be nearly flat in the stagger, so it validates the\n');
fprintf('angle implementation but carries almost no information about the value.\n');
fprintf('The fit is driven by the range, and by HOW the range improves from P=4 to\n');
fprintf('P=12: the paper reports 4.5x, which is far steeper than the sqrt(3)=1.73x\n');
fprintf('that sqrt(P) averaging alone can give.  That gap is the reason both\n');
fprintf('parameterisations are tried below.\n');
cfg = luo_config();
best = struct('err', inf, 'val', NaN, 'mode', '');
for md = {'total','step'}
    fprintf('\n--- stagger_mode = ''%s'' ---\n', md{1});
    if strcmp(md{1},'step'), STAG = [0.2 0.5 1 2 4 8];
    else,                    STAG = [0.5 1 2 4 8 16 30 45]; end
    fprintf('%10s | %-22s | %-22s | %8s %8s\n', ...
            'stagger','P=4  (.031/.165)','P=12 (.018/.037)','P4->P12','mismatch');
    for s = STAG
        c2 = CFG; c2.stagger_mode = md{1};
        [t4, r4]  = luo_high_rmse(cfg, 4,  s, c2, 5);
        [t12,r12] = luo_high_rmse(cfg, 12, s, c2, 15);
        e = mean(abs(log([t4/0.031, r4/0.165, t12/0.018, r12/0.037])));
        fprintf('%9.2f deg | %8.4f %12.4f | %8.4f %12.4f | %7.2fx %8.3f\n', ...
                s, t4, r4, t12, r12, r4/r12, e);
        if e < best.err, best.err = e; best.val = s; best.mode = md{1}; end
    end
end
fprintf('\n  Luo2024''s own P=4 -> P=12 range improvement is 0.165/0.037 = 4.46x.\n');
fprintf('  Whichever mode reproduces THAT ratio is the one matching their scheme;\n');
fprintf('  a mode that cannot reach it is the wrong parameterisation, however\n');
fprintf('  well a single anchor happens to fit.\n');
fprintf('\n  BEST: stagger_mode = ''%s'', stagger = %.2f deg (mismatch %.3f)\n', ...
        best.mode, best.val, best.err);
fprintf('  -> set CFG.stagger_mode and CFG.stagger_deg accordingly.\n');
fprintf('  Below ~0.2 means all four anchors within about 20%%.  If neither mode\n');
fprintf('  gets there, report that the paper''s numbers cannot be reproduced from\n');
fprintf('  what it states -- that is a finding, not a failure.\n');
end

function fit_rmid(CFG)
fprintf('\n=== 2. Luo2024 r_mid1 / r_mid2 ===\n');
fprintf('Luo2024 Sec. IV-B only asks for "appropriate" values making xi_2 ~ 2*pi*p,\n');
fprintf('with no rule.  The angle stage is what they control, so they are fitted\n');
fprintf('against the ANGLE anchors alone.\n\n');
cfg = luo_config();
FR = [0.15 0.3 0.5 0.7 0.85];
fprintf('%12s | %14s | %14s | %s\n','r_mid frac','P=4 (.031)','P=12 (.018)','mismatch');
best = struct('err', inf, 'val', NaN);
for f = FR
    c2 = CFG; c2.rmid_frac = f;
    [t4,  ~] = luo_high_rmse(cfg, 4,  CFG.stagger_deg, c2, 5);
    [t12, ~] = luo_high_rmse(cfg, 12, CFG.stagger_deg, c2, 15);
    e = mean(abs(log([t4/0.031, t12/0.018])));
    fprintf('%12.2f | %14.4f | %14.4f | %6.3f\n', f, t4, t12, e);
    if e < best.err, best.err = e; best.val = f; end
end
fprintf('\n  BEST r_mid fraction = %.2f  -> r_mid = %.1f m\n', ...
        best.val, cfg.r_min + best.val*(cfg.r_max-cfg.r_min));
fprintf('  -> set CFG.rmid_frac = %.2f\n', best.val);
end

function fit_grid(CFG)
fprintf('\n=== 3. Search-grid resolution ===\n');
fprintf('Not fitted to an anchor -- just made fine enough that it is not the\n');
fprintf('limit.  Increase until the result stops moving, then stop.\n\n');
cfg = luo_config();
fprintf('%10s | %12s | %12s\n','n_grid','step (m)','RMSE_r');
prev = NaN;
for g = [500 1000 2000 4000 8000 16000]
    c2 = CFG; c2.n_grid = g;
    [~, r] = luo_high_rmse(cfg, 12, CFG.stagger_deg, c2, 15);
    step = (cfg.r_max+10 - max(1,cfg.r_min-5))/g;
    fprintf('%10d | %12.4f | %12.4e', g, step, r);
    if ~isnan(prev), fprintf('   (%+.1f%% vs previous)', 100*(r-prev)/prev); end
    fprintf('\n'); prev = r;
end
fprintf('\n  Take the smallest n_grid past which the change is under a few percent.\n');
end

function sweep_S(CFG)
fprintf('\n=== 4. Zhang2026 |S| -- NOT a fit, a demonstration ===\n');
fprintf('Its Table II omits |S| although the accuracy scales as 1/sqrt(|S|).\n');
fprintf('Its anchor is the claim RMSE_r < 1e-3 m.  No |S| reaches it: even at\n');
fprintf('|S| = M the incoherent bound of its own class is 4.91e-3 m at 20 dB.\n');
fprintf('So use the largest tractable |S| and report it as the most generous\n');
fprintf('assumption available, rather than fitting a value.\n\n');
cfg = zhang_config();
fprintf('%8s | %14s | %16s | %s\n','|S|','RMSE_r (m)','incoherent bound','claim reachable?');
for S = [1 5 16 64 256]
    c2 = CFG; c2.S_sub = S;
    r = zhang_rmse(cfg, 0.0, c2);
    bnd = 4.914e-2 * sqrt(2048/S);           % 0 dB bound, scaled from |S| = M
    fprintf('%8d | %14.4e | %16.4e | %s\n', S, r, bnd, ternary(bnd < 1e-3,'yes','NO'));
end
fprintf('\n  The bound column never falls below 1e-3 m, which is the point.\n');
fprintf('  -> set CFG.S_sub to the largest value you can afford to run.\n');
end

% ============================================================ the two baselines

function [rmse_th, rmse_r] = luo_high_rmse(cfg, P, stagger_deg, CFG, seed)
rng(seed);
et = zeros(CFG.N_TRIAL,1); er = zeros(CFG.N_TRIAL,1);
for t = 1:CFG.N_TRIAL
    [th, r] = cbs_high(cfg, cfg.r_k, cfg.th_k, cfg.snr, P, stagger_deg, CFG);
    et(t) = rad2deg(th - cfg.th_k);  er(t) = r - cfg.r_k;
end
rmse_th = sqrt(mean(et.^2));  rmse_r = sqrt(mean(er.^2));
end

function [th_hat, r_hat] = cbs_high(cfg, r_k, th_k, snr_db, P, stagger_deg, CFG)
% Luo2024 Sec. IV-C.  P staggered sweeps; angle averaged (its Eq. 26); range
% from matching measured against predicted phases (its Eq. 27/28), with the
% predicted phase evaluated EXACTLY -- the sum over antennas moves with r too,
% and dropping it puts the search on the wrong surface entirely.
r_mid = cfg.r_min + CFG.rmid_frac*(cfg.r_max - cfg.r_min);
ths = zeros(P,1); fs = zeros(P,1); phases = zeros(P,1);
phis = zeros(P,cfg.N); taus = zeros(P,cfg.N);
for p = 1:P
    if strcmp(CFG.stagger_mode, 'step')
        pad = deg2rad(stagger_deg)*((p-1) - (P-1)/2);          % span grows with P
    else
        pad = deg2rad(stagger_deg)*((p-1) - (P-1)/2)/max(P-1,1);% span fixed
    end
    th_s = cfg.theta_max + pad;  th_e = cfg.theta_min - pad;
    [tt, ~] = trajectory(cfg, th_s, r_mid, th_e, r_mid);
    [phi, tau] = ttd_ps(cfg, th_s, r_mid, th_e, r_mid);
    z = sweep(cfg, r_k, th_k, phi, tau, snr_db);
    [~, m] = max(abs(z).^2);
    ths(p) = tt(m);  fs(p) = cfg.f(m);  phases(p) = angle(z(m));
    phis(p,:) = phi;  taus(p,:) = tau;
end
th_hat = mean(ths);
% Grid clamped strictly positive: dist_fresnel divides by r, and a grid touching
% zero fills the score with NaN so argmax silently returns the first bin.
grid = linspace(max(1.0, cfg.r_min-5), cfg.r_max+10, CFG.n_grid).';
acc = zeros(CFG.n_grid,1);
for p = 1:P
    d = dist_fresnel(grid, th_hat, cfg.x);
    ph = fs(p)*d/cfg.c0 - phis(p,:) - (fs(p)-cfg.f0)*taus(p,:);
    pred = angle(sum(exp(2i*pi*ph), 2));
    acc = acc + exp(1i*(phases(p) - pred));
end
[~, k] = max(abs(acc));  r_hat = grid(k);
end

function z = sweep(cfg, r_k, th_k, phi, tau, snr_db)
% Noise referred to the ANTENNAS at the same per-element SNR used everywhere.
% Normalising to the sweep's mean received power instead -- the obvious first
% guess -- hands a squint sweep far more noise than a focused scheme sees,
% because most of its subcarriers sit off-target.
rk = dist_fresnel(r_k, th_k, cfg.x);
ph = cfg.f(:)*rk/cfg.c0 - phi - (cfg.f(:)-cfg.f0)*tau;
z = sum(exp(2i*pi*ph), 2);
s = sqrt(cfg.N * 10^(-snr_db/10) / 2);
z = z + s*(randn(size(z)) + 1i*randn(size(z)));
end

function r = zhang_rmse(cfg, snr_db, CFG)
% Zhang2026: squint power peak, then geometry-compensated smoothing + local
% MUSIC over |S| subcarriers fused by geometric mean (its Eq. 44).
rng(7);
er = zeros(CFG.N_TRIAL,1);
[tt, tr] = trajectory(cfg, cfg.theta_min, cfg.r_min, cfg.theta_max, cfg.r_max);
[phi, tau] = ttd_ps(cfg, cfg.theta_min, cfg.r_min, cfg.theta_max, cfg.r_max);
idx = round(linspace(1, numel(cfg.f), CFG.S_sub));
Ms = cfg.N/2;  Psub = cfg.N - Ms + 1;  p0 = floor((Psub-1)/2) + 1;
for t = 1:CFG.N_TRIAL
    z = sweep(cfg, cfg.r_k, cfg.th_k, phi, tau, snr_db);
    [~, m] = max(abs(z).^2);
    th0 = tt(m);  r0 = tr(m);
    logspec = zeros(41,41);
    for ii = idx
        f = cfg.f(ii);
        a = exp(-2i*pi*f*dist_fresnel(cfg.r_k, cfg.th_k, cfg.x)/cfg.c0).';
        sg = sqrt(10^(-snr_db/10)/2);
        y = a + sg*(randn(cfg.N,1) + 1i*randn(cfg.N,1));
        dref = dist_fresnel(r0, th0, cfg.x(p0:p0+Ms-1));
        R = zeros(Ms);
        for p = 1:Psub
            dp = dist_fresnel(r0, th0, cfg.x(p:p+Ms-1));
            D = exp(-2i*pi*f*(dref-dp)/cfg.c0).';
            yt = D .* y(p:p+Ms-1);
            R = R + yt*yt';
        end
        [V,Dg] = eig(R/Psub);  [~,o] = sort(real(diag(Dg)),'descend');  u1 = V(:,o(1));
        gth = linspace(th0-deg2rad(1), th0+deg2rad(1), 41);
        gr  = linspace(r0-1, r0+1, 41);
        for iy = 1:41
            dd = dist_fresnel(gr(:), gth(iy), cfg.x(p0:p0+Ms-1));
            A = exp(-2i*pi*f*dd/cfg.c0);
            den = Ms - abs(A*conj(u1)).^2;
            logspec(iy,:) = logspec(iy,:) - log(max(den,1e-30)).';
        end
    end
    [~, k] = max(logspec(:));  [~, jr] = ind2sub([41 41], k);
    gr = linspace(r0-1, r0+1, 41);
    er(t) = gr(jr) - cfg.r_k;
end
r = sqrt(mean(er.^2));
end

% ============================================================ shared model

function c = luo_config()
% Luo2024 Sec. V-C, the configuration behind its Fig. 13.
c = base_config(128, 1e9, 2048);
c.r_min = 5; c.r_max = 80;
c.theta_min = deg2rad(-60); c.theta_max = deg2rad(60);
c.r_k = 60; c.th_k = deg2rad(20); c.snr = 10;
end

function c = zhang_config()
% Zhang2026 Table II.
c = base_config(256, 3e9, 2048);
c.r_min = 15; c.r_max = 50;
c.theta_min = deg2rad(-60); c.theta_max = deg2rad(60);
c.r_k = 30; c.th_k = deg2rad(15); c.snr = 0;
end

function c = base_config(N, W, M)
c.c0 = 299792458; c.N = N; c.fc = 60e9; c.W = W; c.M = M;
c.lambda = c.c0/c.fc; c.d = c.lambda/2; c.f0 = c.fc - W/2;
c.fM = c.fc + W/2; c.D = (N-1)*c.d;
c.x = ((0:N-1) - (N-1)/2)*c.d;
c.f = c.f0 + (0:M-1)*(W/M);
end

function [th, r] = trajectory(cfg, th_s, r_s, th_e, r_e)
ft = cfg.f(:) - cfg.f0;  f = cfg.f(:);
ws = (cfg.W - ft).*cfg.f0 ./ (cfg.W*f);
we = (cfg.W + cfg.f0).*ft ./ (cfg.W*f);
sn = min(max(ws*sin(th_s) + we*sin(th_e), -1), 1);
th = asin(sn);
r = 1 ./ ((ws*cos(th_s)^2/r_s + we*cos(th_e)^2/r_e) ./ cos(th).^2);
end

function [phi, tau] = ttd_ps(cfg, th_s, r_s, th_e, r_e)
phi = cfg.f0 * dist_fresnel(r_s, th_s, cfg.x) / cfg.c0;
tau = cfg.fM * dist_fresnel(r_e, th_e, cfg.x) / (cfg.W*cfg.c0) - phi/cfg.W;
end

function d = dist_fresnel(r, th, x)
r = r(:);  x = x(:).';
d = r - x*sin(th) + (x.^2)*cos(th)^2./(2*r);
end

function o = ternary(c, a, b)
if c, o = a; else, o = b; end
end
