function scaling_law()
% SCALING_LAW  Where does near-field range information actually come from?
%
% Self-contained MATLAB counterpart of sim/scaling_law.py.  Run with no
% arguments; it prints the exponent table, the closed-form ratio check, and
% the crossover range, then draws two figures.
%
% Two mutually exclusive sources of range information exist in this system.
%
%   CURVATURE  - what Luo2024, Zhang2026 and arXiv:2603.16390 all use.
%     Range enters only through the quadratic Fresnel term x^2 cos^2(th)/(2r).
%     With the complex amplitude alpha unknown, the constant part of dPhi/dr is
%     absorbed by arg(alpha), leaving only the deviation of x^2 about its mean.
%     For x uniform on [-D/2, D/2], sum_n (x^2-<x^2>)^2 = N D^4/180, hence
%         J_r^curv  = 2 SNR (2 pi fc/c)^2 cos^4(th) N D^4 / (720 r^4)
%
%   DELAY  - what nobody in this lineage uses.
%     Phi(n,m) = -(2 pi f_m/c) r_n is exactly linear in m, with no near-field
%     approximation of any kind.  One shared alpha absorbs only the mean over
%     frequency, leaving the deviation f_m - fc, hence
%         J_r^delay = 2 SNR N M (2 pi/c)^2 W^2 / 12
%
% Therefore
%     RMSE_r^curv  ~ lambda r^2 / (N^2.5 d^2 cos^2 th)     -- grows as r^2
%     RMSE_r^delay ~ c / (W sqrt(N M))                     -- flat in r and th
%     ratio        = sqrt(60 M) (W/fc) r^2 / (D^2 cos^2 th)
%
% Zhang2026 fuses subcarriers incoherently (its Eq. 44), buying sqrt(M) of
% averaging but no delay term, so it lands a factor sqrt(M) below the curvature
% bound and still sqrt(60)(W/fc) r^2/(D^2 cos^2 th) above the delay bound.

clc;
cfg = config();
r0 = 30.0;  th0 = deg2rad(15.0);  snr = 0.0;

fprintf('reference: N=%d, M=%d, W=%.0f GHz, fc=%.0f GHz, D=%.4f m, SNR=%.0f dB/element\n\n', ...
    cfg.N, cfg.M, cfg.W/1e9, cfg.fc/1e9, cfg.D, snr);

%% 1. Exponents
fprintf('%s\n1. EXPONENTS  (numerically fitted vs analytically predicted)\n%s\n', ...
    repmat('=',1,78), repmat('=',1,78));

Ns = [64 128 256 512 1024];
cv = arrayfun(@(n) rmse_curv (setfield(config(),'N',n), r0, th0, snr), Ns); %#ok<*SFLD>
dl = arrayfun(@(n) rmse_delay(setfield(config(),'N',n), r0, th0, snr, 256), Ns);
rows = {'RMSE_curv  vs N', slope(Ns,cv), -2.5; 'RMSE_delay vs N', slope(Ns,dl), -0.5};

rs = [10 20 30 50 80];
rows(end+1,:) = {'RMSE_curv  vs r', slope(rs, arrayfun(@(r) rmse_curv (cfg,r,th0,snr), rs)), 2.0};
rows(end+1,:) = {'RMSE_delay vs r', slope(rs, arrayfun(@(r) rmse_delay(cfg,r,th0,snr,256), rs)), 0.0};

Ws = [0.5 1 2 3 6]*1e9;
rows(end+1,:) = {'RMSE_delay vs W', slope(Ws, arrayfun(@(w) rmse_delay(rebuild(setfield(config(),'W',w)), r0,th0,snr,256), Ws)), -1.0};

Ms = [256 512 1024 2048];
rows(end+1,:) = {'RMSE_delay vs M', slope(Ms, arrayfun(@(m) rmse_delay(rebuild(setfield(config(),'M',m)), r0,th0,snr,128), Ms)), -0.5};
rows(end+1,:) = {'RMSE_incoh vs M', slope(Ms, arrayfun(@(m) rmse_incoh(rebuild(setfield(config(),'M',m)), r0,th0,snr,128), Ms)), -0.5};

ths = deg2rad([5 15 30 45 55]);
rows(end+1,:) = {'RMSE_curv vs 1/cos^2', slope(1./cos(ths).^2, arrayfun(@(t) rmse_curv(cfg,r0,t,snr), ths)), 1.0};

fprintf('%-22s%10s%12s\n','quantity','fitted','predicted');
for i = 1:size(rows,1)
    ok = 'OK'; if abs(rows{i,2}-rows{i,3}) > 0.06, ok = 'MISMATCH'; end
    fprintf('%-22s%10.3f%12.1f   %s\n', rows{i,1}, rows{i,2}, rows{i,3}, ok);
end

%% 2. Closed-form ratio
fprintf('\n%s\n2. THE CLOSED-FORM RATIO  vs numerically evaluated bounds\n%s\n', ...
    repmat('=',1,78), repmat('=',1,78));
fprintf('%-16s%12s%12s%11s%11s%7s\n','(r, theta)','RMSE_curv','RMSE_delay','measured','predicted','err');
for r = [10 30 50 80]
    for thd = [15 45]
        th = deg2rad(thd);
        a = rmse_curv(cfg,r,th,snr);  b = rmse_delay(cfg,r,th,snr,256);
        meas = a/b;  pred = pred_ratio(cfg,r,th);
        fprintf('(%2.0f m, %2.0f deg)   %12.3e%12.3e%11.0f%11.0f%6.1f%%\n', ...
            r, thd, a, b, meas, pred, 100*abs(meas-pred)/pred);
    end
end

%% 3. Crossover
fprintf('\n%s\n3. CROSSOVER: beyond what range is curvature ranging already hopeless?\n%s\n', ...
    repmat('=',1,78), repmat('=',1,78));
for thd = [0 15 30 45 60]
    rx = cross_range(cfg, deg2rad(thd));
    fprintf('  theta = %4.0f deg  ->  curvature competitive only within r < %.1f cm\n', thd, rx*100);
end
fprintf('\n  Zhang2026 declares its sensing region to start at %.0f m, which is %.0fx\n', ...
    15, 15/cross_range(cfg,0));
fprintf('  beyond that crossover.\n');

%% Figures
rr = linspace(1, 80, 200);
figure('Position',[100 100 1100 420]);
subplot(1,2,1);
loglog(rr, arrayfun(@(r) rmse_curv (cfg,r,th0,snr), rr), 'LineWidth', 1.8); hold on;
loglog(rr, arrayfun(@(r) rmse_incoh(cfg,r,th0,snr,256), rr), '--', 'LineWidth', 1.8);
loglog(rr, arrayfun(@(r) rmse_delay(cfg,r,th0,snr,256), rr), '-.', 'LineWidth', 1.8);
grid on; xlabel('range r (m)'); ylabel('RMSE_r bound (m)');
legend({'curvature only (narrowband)','incoherent wideband (Zhang2026 class)','coherent delay (proposed)'}, ...
    'Location','northwest','FontSize',8);
title('(a) Range bound vs distance,  \theta = 15\circ');

subplot(1,2,2);
NN = round(logspace(log10(32), log10(2048), 24));
loglog(NN, arrayfun(@(n) rmse_curv (rebuild(setfield(config(),'N',n)), r0,th0,snr), NN), 'LineWidth', 1.8); hold on;
loglog(NN, arrayfun(@(n) rmse_delay(rebuild(setfield(config(),'N',n)), r0,th0,snr,256), NN), '-.', 'LineWidth', 1.8);
grid on; xlabel('number of antennas N'); ylabel('RMSE_r bound (m)');
legend({'curvature only  (slope -2.5)','coherent delay  (slope -0.5)'}, 'Location','southwest','FontSize',8);
title('(b) Range bound vs array size,  r = 30 m');
end

% ---------------------------------------------------------------- helpers

function c = config()
c.N = 256; c.fc = 60e9; c.W = 3e9; c.M = 2048;
c = rebuild(c);
end

function c = rebuild(c)
c.c0 = 299792458;
c.lambda = c.c0/c.fc;
c.d = c.lambda/2;
c.f0 = c.fc - c.W/2;
c.D = (c.N-1)*c.d;
c.x = ((0:c.N-1) - (c.N-1)/2)*c.d;
c.freqs = c.f0 + (0:c.M-1)*(c.W/c.M);
end

function d = dist_fresnel(r, th, x)
d = r - x*sin(th) + x.^2*cos(th)^2/(2*r);
end

function [bth, br] = crlb_nb(cfg, r, th, snr_db, marg)
% Narrowband bound at fc.  marg=true marginalises the unknown complex alpha,
% which is what Zhang2026 Appendix B declares but never actually does.
k = 2*pi*cfg.fc/cfg.c0;  x = cfg.x(:);
dth = k*(x*cos(th) - x.^2*sin(2*th)/(2*r));
dr  = -k*(1 - x.^2*cos(th)^2/(2*r^2));
a   = exp(-1j*2*pi*cfg.fc*dist_fresnel(r,th,x)/cfg.c0);
D   = [a.*1j.*dth, a.*1j.*dr];
if marg, D = D - a*((a'*D)/(a'*a)); end
J   = 2*10^(snr_db/10)*real(D'*D);
Ci  = inv(J);  bth = sqrt(Ci(1,1));  br = sqrt(Ci(2,2));
end

function [bth, br] = crlb_wb(cfg, r, th, snr_db, nsub, coherent)
% Wideband bound.  coherent=true  -> one alpha shared across subcarriers, so
%                                    the phase-vs-frequency slope is observable.
% coherent=false -> an independent alpha_m per subcarrier, which is the limit
%                   of per-subcarrier processing such as Zhang2026 Eq. (41)/(44).
step = max(1, floor(cfg.M/nsub));
f = cfg.freqs(1:step:end);  f = f(1:min(nsub,numel(f)));  f = f(:);
scale = cfg.M/numel(f);
x = cfg.x(:).';                                  % 1 x N
k = 2*pi*f/cfg.c0;                               % F x 1
dth = k.*(x*cos(th) - x.^2*sin(2*th)/(2*r));     % F x N
dr  = -k.*(1 - x.^2*cos(th)^2/(2*r^2));
A   = exp(-1j*2*pi*f*dist_fresnel(r,th,x)/cfg.c0);
Dth = A.*1j.*dth;  Dr = A.*1j.*dr;
snr = 10^(snr_db/10);
if coherent
    a = A(:);  D = [Dth(:), Dr(:)];
    D = D - a*((a'*D)/(a'*a));
    J = 2*snr*real(D'*D)*scale;
else
    J = zeros(2);
    for i = 1:numel(f)
        a = A(i,:).';  D = [Dth(i,:).', Dr(i,:).'];
        D = D - a*((a'*D)/(a'*a));
        J = J + 2*snr*real(D'*D);
    end
    J = J*scale;
end
Ci = inv(J);  bth = sqrt(Ci(1,1));  br = sqrt(Ci(2,2));
end

function v = rmse_curv(cfg, r, th, snr),        [~,v] = crlb_nb(cfg,r,th,snr,true);          end
function v = rmse_delay(cfg, r, th, snr, ns),   [~,v] = crlb_wb(cfg,r,th,snr,ns,true);       end
function v = rmse_incoh(cfg, r, th, snr, ns),   [~,v] = crlb_wb(cfg,r,th,snr,ns,false);      end

function s = slope(xs, ys), p = polyfit(log(xs(:)), log(ys(:)), 1); s = p(1); end

function q = pred_ratio(cfg, r, th)
q = sqrt(60*cfg.M)*(cfg.W/cfg.fc)*r^2/(cfg.D^2*cos(th)^2);
end

function rx = cross_range(cfg, th)
rx = cfg.D*cos(th)*sqrt(cfg.fc/cfg.W/sqrt(60*cfg.M));
end
