function [error_pos] = iso_forest(res)
% =====================================================================
%  Isolation Forest Wrapper for Outlier Detection
%  ------------------------------------------------------------
%  This function applies Isolation Forest to detect anomalies in
%  PV feature–power data. Each feature column (with Power) is
%  evaluated separately, and anomaly indices are returned.
%
%  Inputs
%  ------
%  res : numeric matrix [N x D]
%        - Each row: one sample
%        - Last column: power (target variable)
%
%  Outputs
%  -------
%  error_pos : cell array
%        - error_pos{i} contains row indices flagged as anomalies
%          when using column i (feature + power) in Isolation Forest.
%
%  Notes
%  -----
%  • Uses MATLAB `iforest` function (Statistics and ML Toolbox).
%  • Default contaminationFraction = 0.05 (≈5% anomalies).
%  • Visualization (t-SNE + scatter plots) is provided but commented.
%
%  Author: Yue Qi
%  James Watt School of Engineering, University of Glasgow
% =====================================================================

%% Hyperparameter setup
rng("default")                         % Fix random seed for reproducibility
contaminationFraction = single(0.05);  % Expected anomaly proportion (5%)

%% Run Isolation Forest detection
for i = 1:size(res,2)
    data = [res(:,i) res(:,end)];   % Pair: feature i + power
    [forest, tf_forest(:,i), s_forest] = iforest( ...
        data, ContaminationFraction = contaminationFraction);
end

%% Collect anomaly indices
for i = 1:size(res,2)
    error_pos{i,1} = find(tf_forest(:,i) == 1);
end

%% (Optional) Dimensionality reduction with t-SNE
% T = tsne(res, Standardize = true);

%% (Optional) Visualization
% figure
% gscatter(res_new(:, 1), res_new(:, end), tf_forest(:,1), "br", [], 15, "off")
% legend("Normal", "Outlier")
% title("Isolation Forest Detection")
% set(gcf,'color','w')
% 
% % Optionally remove anomalies
% res(error_pos,:) = [];

end
