% =====================================================================
%  PV Dataset Cleaning & Validation Pipeline
%  ------------------------------------------------------------
%  This script cleans a 15-minute PV dataset that has been “reverse-corrupted”
%  (spikes, row anomalies, NaNs), then detects & repairs outliers using:
%    - forward fill for missing values,
%    - sliding median clipping (filloutliers),
%    - robust linear fit between GHI (col 8) and Power (last column)
%      via RANSAC (fallback to robustfit / OLS),
%    - optional isolation forest (if user function iso_forest is available),
%  and finally evaluates/visualizes before-vs-after quality metrics.
%
%  Inputs  : 2021PV15min_noisy.csv (numeric matrix; last column = power)
%  Outputs : res_pre.mat / res_post.mat
%            change_log.csv (+ optional change_log.xlsx)
%            figures (correlation heatmaps, residual histograms, etc.)
%
%  Notes
%  -----
%  • Column semantics used in this script for diagnostics/plots:
%      6 = DHI, 7 = DNI, 8 = GHI, 17 = Zenith, last = P_actual (Power)
%  • All comments and messages are written to match the current code exactly.
% =====================================================================

%% 0. Environment
warning off             % suppress warnings
close all               % close any open figures
clear                   % clear workspace
clc                     % clear command window

%% 1. Read (no cropping)
% NOTE: xlsread can read CSV in MATLAB; we keep all rows/columns.
% res = xlsread('2020TO2024PV15min_noisy.csv');
res = xlsread('2021PV15min_noisy.csv');
res = res(:, :);   % keep all rows/columns
fprintf('Original data shape: %d x %d\n', size(res,1), size(res,2));

%% 2. Drop “all-identical” rows across feature columns (keep power column separate)
% We test only feature columns (all but the last power column).
same_pos = [];
for i = 1:size(res,1)
    row = res(i,1:end-1);        % feature columns only (last column is power)
    if all(row == row(1))
        same_pos = [same_pos; i];
    end
end
if ~isempty(same_pos)
    res(same_pos,:) = [];
    fprintf('Removed identical rows: %d\n', numel(same_pos));
end

%% 3. Forward-fill missing values (protect first row)
% If row 1 has NaNs, fill with column means; for subsequent rows, forward fill.
missing = isnan(res);
if any(missing(:))
    if any(isnan(res(1,:)))
        colMean = mean(res,'omitnan');
        nanIdx = isnan(res(1,:));
        res(1, nanIdx) = colMean(nanIdx);
    end
    for i=2:size(res,1)
        nanIdx = isnan(res(i,:));
        res(i, nanIdx) = res(i-1, nanIdx);
    end
end

%% 4. Clip outliers via moving median (window = 15)
% Apply filloutliers to all columns using a moving median window.
[B,TF,L,U,C] = filloutliers(res, "clip", "movmedian", 15);
res_clip = B;
fprintf('Outlier ratio (filloutliers flagged): %.2f%%\n', 100*mean(TF(:)));

% Snapshot after basic pre-clean (before RANSAC / isolation forest replacement)
res_pre = res_clip;

%% 5. Correlation heatmap (BEFORE): columns [6,7,8,17,last]
% Indices: 6=DHI, 7=DNI, 8=GHI, 17=Zenith, last=P_actual.
selCols  = [6,7,8,17, size(res_pre,2)];
colNames = {'DHI','DNI','GHI','Zenith','P_{actual}'};

R1 = corrcoef(res_pre(:, selCols));
figure('Name','Correlation Heatmap (Before)');
imagesc(R1); axis image; colorbar;
title('Correlation Heatmap (Before)');
set(gca,'XTick',1:numel(selCols),'YTick',1:numel(selCols), ...
    'XTickLabel',colNames,'YTickLabel',colNames);

%% 6. RANSAC/robust linear fit on GHI (col 8) vs Power (last col)
% Priority: RANSAC -> robustfit -> OLS. Residuals are based on y - (k*x+b).
irr_col = 8;                    % GHI column
x = res_pre(:, irr_col);        % irradiance
y = res_pre(:, end);            % power

use_ransac   = exist('ransac','file') == 2;
sampleSize   = 25;
maxDistance  = 350;

% RANSAC helpers (linear model: y = k*x + b)
fitLineFcn  = @(xyPoints) polyfit(xyPoints(:,1), xyPoints(:,2), 1);
evalLineFcn = @(model, xyPoints) sum((xyPoints(:,2) - polyval(model, xyPoints(:,1))).^2, 2);

xyPoints = [x y];

if use_ransac
    try
        [modelRANSAC, inlierIdx] = ransac(xyPoints, fitLineFcn, evalLineFcn, sampleSize, maxDistance);
    catch
        warning('RANSAC failed; falling back to robustfit.');
        use_ransac = false;
    end
end

if ~use_ransac
    if exist('robustfit','file') == 2
        [b,~] = robustfit(x, y);            % y = b0 + b1*x
        modelRANSAC = [b(2) b(1)];          % convert to [k b]
        resid = abs(y - (modelRANSAC(1)*x + modelRANSAC(2)));
        inlierIdx = resid <= prctile(resid, 75);
    else
        modelRANSAC = polyfit(x,y,1);
        resid = abs(y - polyval(modelRANSAC, x));
        inlierIdx = resid <= prctile(resid, 75);
    end
end

% Visualize inliers/outliers and the fitted line
figure('Name','RANSAC / Robust Fit & Inliers/Outliers');
plot(x(inlierIdx), y(inlierIdx), '.', 'DisplayName','Inliers'); hold on;
plot(x(~inlierIdx), y(~inlierIdx), 'ro', 'DisplayName','Outliers');
x2 = linspace(min(x), max(x), 200);
y2 = polyval(modelRANSAC, x2);
plot(x2, y2, 'g-', 'LineWidth',1.5, 'DisplayName','RANSAC/Robust Fit');
grid on; legend('Location','NorthWest');
xlabel('GHI'); ylabel('Power'); title('GHI–Power: Robust/RANSAC Fit');

%% 7. Theoretical power from fit & residuals
% Negative theoretical power is clipped to 0; residuals feed anomaly detection.
T_linear = modelRANSAC(1)*x + modelRANSAC(2);
T_linear(T_linear<0) = 0;
residual_power = abs(y - T_linear);

%% 8. Isolation Forest second-stage anomaly detection (fallback to isoutlier)
% Features = [DHI(6), DNI(7), GHI(8), Zenith(17), residual_power].
err_idx_isof = false(size(y));
if exist('iso_forest','file') == 2
    try
        features = res_pre(:, [6, 7, 8, 17]);
        [error_pos2] = iso_forest([features residual_power]);   % user-provided function
        if iscell(error_pos2)
            for k = 1:numel(error_pos2)
                err_idx_isof(error_pos2{k}) = true;
            end
        elseif islogical(error_pos2) || isnumeric(error_pos2)
            err_idx_isof(error_pos2) = true;
        end
    catch
        warning('iso_forest failed; falling back to isoutlier(movmedian) on residual_power.');
        err_idx_isof = isoutlier(residual_power, 'movmedian', 15);
    end
else
    err_idx_isof = isoutlier(residual_power, 'movmedian', 15);
end

%% 9. Replacement policy: if (RANSAC outlier) OR (Isolation-Forest anomaly) → use T_linear
res_post = res_pre;                      % copy
replace_idx = (~inlierIdx) | err_idx_isof;
res_post(replace_idx, end) = T_linear(replace_idx);

%% 10. Correlation heatmap (AFTER) using the same columns as step 5
R2 = corrcoef(res_post(:, selCols));
figure('Name','Correlation Heatmap (After)');
imagesc(R2); axis image; colorbar;
title('Correlation Heatmap (After)');
set(gca,'XTick',1:numel(selCols),'YTick',1:numel(selCols), ...
    'XTickLabel',colNames,'YTickLabel',colNames);

%% 11. Evaluation: before vs after (fit/association/distribution/consistency)
% Metrics: Spearman rho, RMSE/MAE, inlier ratio, KS stats, autocorr (lag1/lag96),
%          negative power ratio, over-rating ratio, high-irradiance/near-zero-power,
%          energy bias, and replacement ratio.
x_pre  = res_pre(:, irr_col);   y_pre  = res_pre(:, end);
x_post = res_post(:, irr_col);  y_post = res_post(:, end);

% Spearman correlation
[rho_pre,  p_pre]  = corr(x_pre,  y_pre,  'Type','Spearman','Rows','complete');
[rho_post, p_post] = corr(x_post, y_post, 'Type','Spearman','Rows','complete');

% Fit errors
mdl_pre  = polyfit(x_pre,  y_pre, 1);
mdl_post = polyfit(x_post, y_post, 1);
yhat_pre  = polyval(mdl_pre,  x_pre);
yhat_post = polyval(mdl_post, x_post);
rmse_pre  = sqrt(mean((y_pre  - yhat_pre ).^2));
rmse_post = sqrt(mean((y_post - yhat_post).^2));
mae_pre   = mean(abs(y_pre  - yhat_pre ));
mae_post  = mean(abs(y_post - yhat_post));

% Inlier ratio (RANSAC if available; otherwise robust/OLS proxy)
[inlier_ratio_pre, inlier_ratio_post] = deal(NaN,NaN);
try
    if exist('ransac','file') == 2
        [~, in_pre ] = ransac([x_pre  y_pre ], fitLineFcn, evalLineFcn, sampleSize, maxDistance);
        [~, in_post] = ransac([x_post y_post], fitLineFcn, evalLineFcn, sampleSize, maxDistance);
        inlier_ratio_pre  = mean(in_pre);
        inlier_ratio_post = mean(in_post);
    else
        [bpre,~] = robustfit(x_pre, y_pre);
        rpre = abs(y_pre - (bpre(2)*x_pre + bpre(1)));
        inlier_ratio_pre  = mean(rpre <= prctile(rpre,75));
        [bpos,~] = robustfit(x_post, y_post);
        rpos = abs(y_post - (bpos(2)*x_post + bpos(1)));
        inlier_ratio_post = mean(rpos <= prctile(rpos,75));
    end
catch
    rpre = abs(y_pre - polyval(polyfit(x_pre, y_pre, 1), x_pre));
    rpos = abs(y_post - polyval(polyfit(x_post,y_post,1), x_post));
    inlier_ratio_pre  = mean(rpre <= prctile(rpre,75));
    inlier_ratio_post = mean(rpos <= prctile(rpos,75));
end

% KS statistics (kstest2 returns [h,p,ksstat]); we log ksstat as a shift proxy
[~,~,ks_power] = kstest2(y_pre,  y_post);
[~,~,ks_irr  ] = kstest2(x_pre,  x_post);

% Autocorrelation (lag1 / lag≈96; reduce if dataset is short)
acf_lag = @(s,lag) corr(s(1+lag:end), s(1:end-lag),'Rows','complete');
lag1_pre  = acf_lag(y_pre,  1);   lag1_post  = acf_lag(y_post, 1);
lag96 = min(96, floor(size(res_post,1)/10));
lag96_pre = acf_lag(y_pre,  lag96);
lag96_post= acf_lag(y_post, lag96);

% Physical consistency indicators
prct = @(a) 100*a;
neg_power_pre  = prct(mean(y_pre  < 0));
neg_power_post = prct(mean(y_post < 0));

Pr_pre  = prctile(y_pre,95);  Pr_post = prctile(y_post,95);
over_rated_pre  = prct(mean(y_pre  > 1.05*Pr_pre ));
over_rated_post = prct(mean(y_post > 1.05*Pr_post));

ghi = res_pre(:,8);
v95 = prctile(ghi,95);
hiI_zeroP_pre  = prct(mean(ghi  > v95 & y_pre  < 0.02*Pr_pre ));
hiI_zeroP_post = prct(mean(ghi  > v95 & y_post < 0.02*Pr_post));

% Energy bias & replacement ratio
energy_bias   = (sum(y_post) - sum(y_pre)) / max(1e-9,sum(y_pre));
changed_ratio = prct(mean(abs(y_post - y_pre) > 1e-9));

% Report
fprintf('\n==== Cleaning Effectiveness Report ====\n');
fprintf('Spearman rho  : before=%.3f (p=%.1e) | after=%.3f (p=%.1e)\n', rho_pre,p_pre, rho_post,p_post);
fprintf('RMSE / MAE    : RMSE %.2f -> %.2f | MAE %.2f -> %.2f\n', rmse_pre,rmse_post, mae_pre,mae_post);
fprintf('Inlier ratio  : %.1f%% -> %.1f%%\n', 100*inlier_ratio_pre, 100*inlier_ratio_post);
fprintf('KS (power/irr): power=%.3f | irr=%.3f (irr should be ≈0)\n', ks_power, ks_irr);
fprintf('ACF(lag1/%d)  : lag1 %.3f -> %.3f | lag%d %.3f -> %.3f\n', ...
        lag96, lag1_pre, lag1_post, lag96, lag96_pre, lag96_post);
fprintf('Negative power: %.2f%% -> %.2f%%\n', neg_power_pre, neg_power_post);
fprintf('Over-rating   : %.2f%% -> %.2f%%\n', over_rated_pre, over_rated_post);
fprintf('Hi-irr, ~0 P  : %.2f%% -> %.2f%%\n', hiI_zeroP_pre, hiI_zeroP_post);
fprintf('Energy bias   : %.2f%%  (recommended |bias| ≤ 5–10%%)\n', 100*energy_bias);
fprintf('Replaced rows : %.2f%%\n', changed_ratio);
fprintf('======================================\n');

%% 12. Residual histograms (before vs after)
figure('Name','Residual Histograms (Before vs After)');
subplot(1,2,1); histogram(y_pre - yhat_pre, 50);  title('Before: residuals'); xlabel('Error'); ylabel('Count'); grid on;
subplot(1,2,2); histogram(y_post - yhat_post,50); title('After: residuals');  xlabel('Error'); ylabel('Count'); grid on;

%% 13. Save intermediate matrices
save('res_pre.mat',  'res_pre');   % before replacement (post pre-clean)
save('res_post.mat', 'res_post');  % after replacement (final clean)
fprintf('Saved res_pre.mat and res_post.mat\n');

%% ==== Old vs New: cell-level diff & export validation ====
% Compare res_pre and res_post cell-wise and export change_log.csv.
if ~exist('res_pre','var') || ~exist('res_post','var')
    error('res_pre / res_post not found. Save res_pre before replacement and res_post after replacement.');
end

% 1) Difference mask (tolerance for floating errors)
EPS  = 1e-9;
mask = abs(res_post - res_pre) > EPS;

% 2) Extract positions and values
[ri, cj] = find(mask);
oldv = res_pre(mask);
newv = res_post(mask);
dv   = newv - oldv;

% 3) Heuristic reason tagging for changes
reason  = repmat("Unknown/Preclean", numel(ri), 1);
lastCol = size(res_post,2);

has_inlier   = exist('inlierIdx','var') == 1;
has_isof     = exist('err_idx_isof','var') == 1;
has_replace  = exist('replace_idx','var') == 1;

for k = 1:numel(ri)
    r = ri(k); c = cj(k);
    if c == lastCol
        tags = string.empty;
        if has_inlier && ~inlierIdx(r), tags(end+1) = "RANSAC"; end
        if has_isof   &&  err_idx_isof(r), tags(end+1) = "iForest"; end
        if isempty(tags)
            if has_replace && replace_idx(r)
                tags = "Replace";
            else
                tags = "Manual/Other";
            end
        end
        reason(k) = strjoin(tags, "+");
    else
        reason(k) = "Preclean(fillmissing/filloutliers)";
    end
end

% 4) Export change log
varNames = {'RowIndex','ColIndex','OldValue','NewValue','Delta','Reason'};
if ~iscellstr(reason)
    reason = cellstr(reason);
end
T = table(ri, cj, oldv, newv, dv, reason, 'VariableNames', varNames);
T = sortrows(T, {'ColIndex','RowIndex'});
writetable(T, 'change_log.csv');
disp('✅ Exported change log: change_log.csv');

% 5) Summary stats of changes
total_cells   = numel(res_post);
changed_cells = height(T);
changed_ratio = changed_cells / total_cells * 100;

col_changes = sum(mask, 1).';
row_changes = sum(mask, 2);

fprintf('\n==== Change Summary ====\n');
fprintf('Changed cells : %d (%.2f%%)\n', changed_cells, changed_ratio);
fprintf('Rows changed  : %d\n', nnz(row_changes));
fprintf('Cols changed  : %d\n', nnz(col_changes));
if any(col_changes)
    fprintf('Per-column changes (col : count):\n');
    for c = 1:numel(col_changes)
        if col_changes(c) > 0
            fprintf('  Col %d : %d\n', c, col_changes(c));
        end
    end
end
fprintf('========================\n');

% 6) Power-column change heatmap (binary)
if any(mask(:,lastCol))
    figure('Name','Power Column Change Heatmap');
    imagesc(mask(:,lastCol));
    colormap(gray); colorbar;
    set(gca,'YDir','normal','XTick',1,'XTickLabel',{'Power'});
    ylabel('Row index');
    title('Power column changed? (1=yes, 0=no)');
end

%% 7) Correlation heatmaps (side-by-side: before vs after)
featNames = {'DHI','DNI','GHI','Zenith','P_{actual}'};

figure('Name','Correlation Heatmaps (Before vs After)','Position',[100 100 1200 500]);

subplot(1,2,1);
imagesc(R1); axis image; colorbar;
title('Correlation Heatmap (Before)');
set(gca,'XTick',1:numel(selCols),'XTickLabel',featNames,...
        'YTick',1:numel(selCols),'YTickLabel',featNames,...
        'XTickLabelRotation',45);

subplot(1,2,2);
imagesc(R2); axis image; colorbar;
title('Correlation Heatmap (After)');
set(gca,'XTick',1:numel(selCols),'XTickLabel',featNames,...
        'YTick',1:numel(selCols),'YTickLabel',featNames,...
        'XTickLabelRotation',45);

sgtitle('PV Feature–Power Correlations (Before vs After)');

%% 8) Optional: also save Excel change log (if supported)
try
    writetable(T, 'change_log.xlsx', 'FileType', 'spreadsheet');
    disp('📄 Also saved: change_log.xlsx');
catch
    % Excel engine not available; ignore.
end

%% 9) Export final cleaned data (CSV)
% res_post is the final cleaned matrix. We export with placeholder headers Var1..VarN.
varNames = compose("Var%d", 1:size(res_post,2));
Tclean   = array2table(res_post, 'VariableNames', varNames);
writetable(Tclean, 'PV_clean.csv');
disp('✅ Saved PV_clean.csv (with headers)');
