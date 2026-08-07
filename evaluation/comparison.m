clear; clc; close all;

load('comparison.mat');

B = B_values(:)';
mimogs_bme = mimogs_BME(:);
baseline_bme = baseline_BME(:);

has_random = exist('random_BME', 'var') && ~isempty(random_BME) && all(isfinite(random_BME(:)));
if has_random
    random_bme = random_BME(:);
end

figure('Color', 'w');
plot(B, mimogs_bme, '-o', 'LineWidth', 2.2, 'MarkerSize', 7); hold on;
plot(B, baseline_bme, '--s', 'LineWidth', 2.2, 'MarkerSize', 7);
if has_random
    plot(B, random_bme, ':^', 'LineWidth', 2.0, 'MarkerSize', 7);
end
grid on;
xlabel('Consumed time slots B');
ylabel('Beam Management Efficiency (BME)');
title('MIMO-GS vs. Baseline Beam Management');
if has_random
    legend({'MIMO-GS', 'Statistical prior baseline', 'Random baseline'}, 'Location', 'best');
else
    legend({'MIMO-GS', 'Statistical prior baseline'}, 'Location', 'best');
end
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 1.2);
box on;
xlim([min(B), max(B)]);

if has_random
    ymax = max([mimogs_bme; baseline_bme; random_bme]);
else
    ymax = max([mimogs_bme; baseline_bme]);
end
ymax = max(ymax, 1e-6);
ylim([0, ymax * 1.1]);

exportgraphics(gcf, 'comparison_BME.png', 'Resolution', 300);
savefig(gcf, 'comparison_BME.fig');

has_components = ...
    exist('mimogs_alignment_accuracy', 'var') && exist('baseline_alignment_accuracy', 'var') && ...
    exist('mimogs_throughput_ratio', 'var') && exist('baseline_throughput_ratio', 'var') && ...
    exist('overhead', 'var');

if has_components
    figure('Color', 'w');

    subplot(3,1,1);
    plot(B, mimogs_alignment_accuracy(:), '-o', 'LineWidth', 2.0, 'MarkerSize', 6); hold on;
    plot(B, baseline_alignment_accuracy(:), '--s', 'LineWidth', 2.0, 'MarkerSize', 6);
    if exist('random_alignment_accuracy', 'var') && ~isempty(random_alignment_accuracy) && all(isfinite(random_alignment_accuracy(:)))
        plot(B, random_alignment_accuracy(:), ':^', 'LineWidth', 1.8, 'MarkerSize', 6);
        legend({'MIMO-GS', 'Statistical prior baseline', 'Random baseline'}, 'Location', 'best');
    else
        legend({'MIMO-GS', 'Statistical prior baseline'}, 'Location', 'best');
    end
    grid on;
    ylabel('Alignment accuracy');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.1);
    xlim([min(B), max(B)]);

    subplot(3,1,2);
    plot(B, mimogs_throughput_ratio(:), '-o', 'LineWidth', 2.0, 'MarkerSize', 6); hold on;
    plot(B, baseline_throughput_ratio(:), '--s', 'LineWidth', 2.0, 'MarkerSize', 6);
    if exist('random_throughput_ratio', 'var') && ~isempty(random_throughput_ratio) && all(isfinite(random_throughput_ratio(:)))
        plot(B, random_throughput_ratio(:), ':^', 'LineWidth', 1.8, 'MarkerSize', 6);
    end
    grid on;
    ylabel('Throughput ratio');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.1);
    xlim([min(B), max(B)]);

    subplot(3,1,3);
    penalty = 1 - overhead(:);
    plot(B, penalty, '-d', 'LineWidth', 2.0, 'MarkerSize', 6);
    grid on;
    xlabel('Consumed time slots B');
    ylabel('Overhead penalty (1-overhead)');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.1);
    xlim([min(B), max(B)]);

    exportgraphics(gcf, 'comparison_components.png', 'Resolution', 300);
end
