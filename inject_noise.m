% =====================================================================
%  Photovoltaic Dataset "Reverse Corruption" Script
%  Focus: Power output corruption + row-level anomalies/missing values
%
%  Requirements:
%   - Output CSV headers must match the original dataset (fixed as expectedHeaders below)
%
%  Corruption Types:
%   A. Power extreme anomalies (tens to hundreds of times, e.g., 60→6000)
%   B. Row anomalies (all numeric columns set to 99 or -99)
%   C. Power missing values (random NaN)
%   D. Row missing values (all numeric columns set to NaN)
%
%  Outputs:
%   - 2020TO2024PV15min_noisy.csv (headers consistent with the source file)
%   - noisy_change_log.csv (detailed log of all modifications)
% =====================================================================

%% 0. Environment
warning off             % Disable warnings
close all               % Close all open figures
clear                   % Clear workspace variables
clc                     % Clear command window

%% === Fixed headers (must be consistent with source file) ===
expectedHeaders = [ ...
    "Time", ...
    "Air Temperature ", ...% °C
    "Azimuth Angle ", ...% °
    "Cloud Opacity", ...
    "Dew Point Temperature ", ...% °C
    "DHI (Diffuse Horizontal Irradiance) ", ...% W/m²
    "DNI (Direct Normal Irradiance) ", ...% W/m²
    "GHI (Global Horizontal Irradiance) ", ...% W/m²
    "GTI (Fixed Tilt Irradiance) W/m²", ...% W/m²
    "GTI (Tracking Tilt Irradiance)", ...% W/m²
    "Precipitable Water ", ...% cm
    "Relative Humidity ", ...% %
    "Snow Depth ", ...% cm
    "Surface Pressure ", ...% hPa (mb)
    "Wind Direction at 10m ", ...% °
    "Wind Speed at 10m ", ...% m/s
    "Zenith Angle ", ...% °
    "P_actual" ...% kW
];

infile  = '2021PV15min.csv';
outfile = '2021PV15min_noisy.csv';
logfile = 'noisy_change_log.csv';

%% === Corruption parameters (tunable) ===
rng(2025);              % Fix random seed
% A) Power extreme anomalies
rate_extreme = 0.02;    % 2% of rows
spike_min    = -5;      % Minimum scaling factor
spike_max    = 10;      % Maximum scaling factor
% B) Row anomalies (values replaced with 99 / -99)
row_anom_cnt = 50;      
row_anom_vals = [99, -99];
% C) Power missing values
rate_missing_pw = 0.01; % 1% of rows
% D) Row missing values (all numeric columns NaN)
row_missing_cnt = 80;

% (Optional) Small jitter (default disabled)
enable_small_jitter = false;
rate_jitter = 0.10;     % jitter magnitude = 10% of current value

%% === Read CSV (preserve headers) ===
% 'VariableNamingRule','preserve' ensures headers with spaces/symbols remain intact
T = readtable(infile, 'TextType','string', 'VariableNamingRule','preserve');
[nrow,ncol] = size(T);
origHeaders = string(T.Properties.VariableNames);

% Check: must have 18 columns
if ncol ~= numel(expectedHeaders)
    warning('Source file has %d columns, expected 18. Will overwrite with expectedHeaders anyway.', ncol);
end

% Find "Actual Power" column
pIdx = find(origHeaders == "P_actual", 1, 'first');
if isempty(pIdx)
    pIdx = find(contains(origHeaders, "Power"), 1, 'first');
end
if isempty(pIdx)
    % Fallback: use last column
    pIdx = ncol;
    warning('Power column not found, fallback to last column.');
end

% Mark numeric columns
isNumCol = false(1,ncol);
for j=1:ncol
    isNumCol(j) = isnumeric(T.(j)) || isfloat(T.(j));
end

% Ensure Power column is double
pw = T.(pIdx);
if ~isfloat(pw), pw = double(pw); end
T.(pIdx) = pw;
pw_clean = pw;

fprintf('Read %s: %d rows × %d cols; Power column = [%d] %s\n', infile, nrow, ncol, pIdx, T.Properties.VariableNames{pIdx});

%% === Change log containers ===
R = []; C = []; Old = []; New = []; Reason = strings(0,1);

%% === A) Power extreme anomalies ===
k = max(1, round(rate_extreme * nrow));
idx_ext = randperm(nrow, k);
factors = spike_min + (spike_max - spike_min) * rand(k,1);

old = pw(idx_ext);
% For near-zero values, use nonzero median as baseline to generate spikes
baseline = max(median(pw(pw>0), 'omitnan'), 1);
pw_spike = old;
zeroish = abs(old) < 1e-9;
pw_spike(zeroish)  = baseline .* factors(zeroish);
pw_spike(~zeroish) = old(~zeroish) .* factors(~zeroish);

pw(idx_ext) = pw_spike;
[R,C,Old,New,Reason] = addlog(R,C,Old,New,Reason, idx_ext, pIdx, old, pw_spike, ...
    "PowerExtreme×["+string(spike_min)+"~"+string(spike_max)+"]");

%% === B) Row anomalies (99 / -99) ===
m = min(row_anom_cnt, nrow);
if m > 0
    rows = randperm(nrow, m);
    for r = rows
        markVal = row_anom_vals(randi(numel(row_anom_vals)));
        for j = 1:ncol
            if isNumCol(j)
                oldv = T.(j)(r);
                newv = markVal;
                if ~isequaln(oldv, newv)
                    T.(j)(r) = newv;
                    R(end+1,1)   = r; %#ok<AGROW>
                    C(end+1,1)   = j;
                    Old(end+1,1) = toDouble(oldv);
                    New(end+1,1) = toDouble(newv);
                    Reason(end+1,1) = "RowAnomaly("+string(markVal)+")";
                end
            end
        end
    end
end

%% === C) Power missing values (NaN) ===
k = round(rate_missing_pw * nrow);
if k > 0
    idx = randperm(nrow, k);
    old = pw(idx);
    pw(idx) = NaN;
    [R,C,Old,New,Reason] = addlog(R,C,Old,New,Reason, idx, pIdx, old, pw(idx), "PowerMissingNaN");
end

%% === D) Row missing values (NaN) ===
m = min(row_missing_cnt, nrow);
if m > 0
    rows = randperm(nrow, m);
    for r = rows
        for j = 1:ncol
            if isNumCol(j)
                oldv = T.(j)(r);
                newv = NaN;
                if ~isequaln(oldv, newv)
                    T.(j)(r) = newv;
                    R(end+1,1)   = r;
                    C(end+1,1)   = j;
                    Old(end+1,1) = toDouble(oldv);
                    New(end+1,1) = toDouble(newv);
                    Reason(end+1,1) = "RowMissingNaN";
                end
            end
        end
    end
end

%% === (Optional) Small jitter on Power ===
if enable_small_jitter && rate_jitter > 0
    noise = randn(nrow,1) .* (rate_jitter * max(1e-9, abs(pw_clean)));
    idx = true(nrow,1);
    old = pw(idx);
    pw(idx) = pw(idx) + noise(idx);
    [R,C,Old,New,Reason] = addlog(R,C,Old,New,Reason, find(idx), pIdx, old, pw(idx), ...
        "Jitter("+num2str(rate_jitter*100,'%.1f')+"%)");
end

%% === Write back Power column ===
T.(pIdx) = pw;

%% === Force output headers consistent with expectedHeaders ===
if numel(expectedHeaders) ~= width(T)
    error('expectedHeaders length (%d) does not match table width (%d).', ...
        numel(expectedHeaders), width(T));
end
T.Properties.VariableNames = cellstr(expectedHeaders);

%% === Save corrupted CSV ===
writetable(T, outfile, 'WriteVariableNames', true);
fprintf('✅ Saved corrupted dataset: %s\n', outfile);

%% === Save modification log ===
Delta = New - Old;
colName = T.Properties.VariableNames(C).';
Log = table(R, C, colName, Old, New, Delta, Reason, ...
    'VariableNames', {'RowIndex','ColIndex','ColName','OldValue','NewValue','Delta','Reason'});
Log = sortrows(Log, {'ColIndex','RowIndex'});
writetable(Log, logfile);
fprintf('📝 Exported change log: %s (%d modifications)\n', logfile, height(Log));

%% === Quick visual checks: Original vs Noisy ===
Por  = readmatrix('2021PV15min.csv');        % Original
Pnoi = readmatrix('2021PV15min_noisy.csv');  % Noisy

n = min(size(Por,1), size(Pnoi,1));
Por  = Por(1:n,:);
Pnoi = Pnoi(1:n,:);

pCol = size(Por,2);

OriginalColor = [47,129,183]/255;   % blue
NoisyColor    = [201,33,26]/255;    % red

% Figure 1: Original Power
figure;
plot(Por(:,pCol), 'Color', OriginalColor, 'LineWidth',1.2);
xlabel('Sample index'); ylabel('Power');
title('Original Power (full dataset)'); grid on;

% Figure 2: Noisy Power
figure;
plot(Pnoi(:,pCol), 'Color', NoisyColor, 'LineWidth',1.2);
xlabel('Sample index'); ylabel('Power');
title('Noisy Power (full dataset)'); grid on;

% Figure 3: Overlay comparison
figure;
tol = 1e-6;
mask_diff = abs(Pnoi(:,pCol) - Por(:,pCol)) > tol;
plot(Por(:,pCol), 'Color', OriginalColor, 'LineWidth',1.2, 'DisplayName','Original'); hold on;
plot(find(mask_diff), Pnoi(mask_diff,pCol), '.', 'Color', NoisyColor, 'MarkerSize',8, 'DisplayName','Noisy');
xlabel('Sample index'); ylabel('Power');
title('Original vs Noisy Power (differences highlighted)');
legend; grid on;

% Figure 4: Boxplot
figure;
boxplot([Por(:,pCol), Pnoi(:,pCol)], ...
        'Labels', {'Original','Noisy'}, 'Whisker',1.5, ...
        'Colors', [OriginalColor; NoisyColor]);
ylabel('Power'); title('Boxplot: Original vs Noisy Power'); grid on;

% Figure 5: Zoom-in comparison
figure;
seg_len = round(0.1 * n);                
start_idx = round(0.45 * n);             
idx_range = start_idx:(start_idx+seg_len-1);

plot(idx_range, Por(idx_range,pCol), 'Color', OriginalColor, 'LineWidth',1.2, 'DisplayName','Original'); hold on;
mask_diff_seg = abs(Pnoi(idx_range,pCol) - Por(idx_range,pCol)) > tol;
plot(idx_range(mask_diff_seg), Pnoi(idx_range(mask_diff_seg),pCol), '.', ...
    'Color', NoisyColor, 'MarkerSize',8, 'DisplayName','Noisy');
xlabel('Sample index'); ylabel('Power');
title(sprintf('Zoom-in view (10%% segment)')); legend; grid on;

%% === Utility functions ===
function [R,C,Old,New,Reason] = addlog(R,C,Old,New,Reason, rows, col, oldv, newv, why)
rows = rows(:); oldv = oldv(:); newv = newv(:);
R   = [R; rows];
C   = [C; repmat(col, numel(rows),1)];
Old = [Old; toDouble(oldv)];
New = [New; toDouble(newv)];
if isstring(why)
    Reason = [Reason; repmat(why, numel(rows),1)];
else
    Reason = [Reason; strings(numel(rows),1) + string(why)];
end
end

function x = toDouble(v)
if isnumeric(v) || islogical(v)
    x = double(v);
else
    try x = str2double(v); catch, x = NaN; end
end
end